import os
import json
import re
import hashlib
from urllib.parse import urlparse

import faiss
from dotenv import load_dotenv
from ddgs import DDGS
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# IMPORTANT:
# Set HF_MODEL_ID to a model that is actually enabled for your
# Hugging Face account. If it is unavailable, the pipeline will
# automatically use Grok as the Tier 2 fallback.
#
# Example:
# HF_MODEL_ID=your_supported_huggingface_chat_model
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "openai/gpt-oss-20b").strip()

# Current xAI API model. The xAI documentation currently recommends
# Grok 4.6 for general text/chat workloads.
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-20b")

# Search settings
MAX_SEARCH_RESULTS = 5

# Tier 1 semantic threshold.
# FAISS uses normalized embeddings, so lower distance = closer match.
TIER1_DISTANCE_THRESHOLD = 0.35

# A semantic match alone is NOT enough to reuse a previous verdict.
# We ask the primary/fallback LLM to compare the new claim with the
# cached claim before reusing its result.
TIER1_REUSE_CONFIDENCE = 85

# Tier 2 confidence weights
TIER2_WEIGHTS = {
    "evidence_strength": 45,
    "source_credibility": 25,
    "llm_certainty": 30
}


class TextVerificationPipeline:
    def __init__(self):
        print("[*] Initializing TruthLens Text Verification Pipeline...")

        if not HF_API_TOKEN and not GROQ_API_KEY:
            raise ValueError(
                "No cloud API key found. Add HF_API_TOKEN and/or "
                "XAI_API_KEY to your .env file."
            )

        # --------------------------------------------------------
        # Local embedding model
        # --------------------------------------------------------
        print("[*] Loading local embedding model...")

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu",
            token=os.getenv("HF_API_TOKEN")
        )

        self.embedding_dim = self.embedder.get_embedding_dimension()

        # Normalized vectors + IndexFlatL2
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

        # Session-level verified claim cache.
        # No database is used.
        self.cache_metadata = []

        self._build_seed_cache()

        # --------------------------------------------------------
        # Hugging Face primary LLM
        # --------------------------------------------------------
        self.hf_client = None

        if HF_API_TOKEN and HF_MODEL_ID:
            print(
                f"[*] Hugging Face primary model configured: "
                f"{HF_MODEL_ID}"
            )

            try:
                self.hf_client = InferenceClient(
                    provider="auto",
                    api_key=HF_API_TOKEN
                )
            except Exception as e:
                print(f"[!] Hugging Face initialization failed: {e}")
                self.hf_client = None
        elif HF_API_TOKEN and not HF_MODEL_ID:
            print(
                "[!] HF_API_TOKEN found but HF_MODEL_ID is not set. "
                "Hugging Face will be skipped."
            )

        # --------------------------------------------------------
        # Groq Console fallback LLM
        # --------------------------------------------------------
        self.groq_client = None

        if GROQ_API_KEY:
            print(
                f"[*] Groq Console fallback configured: "
                f"{GROQ_MODEL_ID}"
            )

            # Groq provides an OpenAI-compatible API.
            self.groq_client = OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
        else:
            print(
                "[!] GROQ_API_KEY not found. "
                "Groq fallback is disabled."
            )

        print("[*] Pipeline Ready!\n")

    # ============================================================
    # Text normalization
    # ============================================================

    @staticmethod
    def _normalize_claim(query):
        """
        Normalize superficial differences so that:
        'Earth is round.'
        and
        '  Earth is round  '
        can share the same session cache.
        """

        query = query.lower().strip()

        # Normalize whitespace
        query = re.sub(r"\s+", " ", query)

        # Remove surrounding punctuation while preserving meaningful
        # punctuation inside the claim.
        query = query.strip(" \t\r\n.,!?;:")

        return query

    @staticmethod
    def _claim_hash(query):
        return hashlib.sha256(
            query.encode("utf-8")
        ).hexdigest()

    # ============================================================
    # Tier 1 seed cache
    # ============================================================

    def _build_seed_cache(self):
        """
        Small seed set for demonstration.

        New high-confidence Tier 2 results are added to this same
        in-memory cache during the current session.
        """

        historical_data = [
            {
                "text": "The Earth is flat and NASA is faking space images.",
                "label": "Fake",
                "confidence_score": "98%",
                "reason": (
                    "The claim contradicts extensive scientific and "
                    "observational evidence."
                ),
                "sources": [
                    {
                        "name": "NASA",
                        "domain": "nasa.gov",
                        "url": "https://www.nasa.gov/",
                        "role": "authoritative reference"
                    }
                ],
                "verification_type": "seed"
            },
            {
                "text": (
                    "Water is composed of two hydrogen atoms and "
                    "one oxygen atom."
                ),
                "label": "Real",
                "confidence_score": "98%",
                "reason": "Established scientific fact.",
                "sources": [
                    {
                        "name": "USGS",
                        "domain": "usgs.gov",
                        "url": "https://www.usgs.gov/",
                        "role": "authoritative reference"
                    }
                ],
                "verification_type": "seed"
            }
        ]

        self._add_cache_entries(historical_data)

    def _add_cache_entries(self, entries):
        """Add normalized claims to the session FAISS cache."""

        if not entries:
            return

        texts = [
            self._normalize_claim(item["text"])
            for item in entries
        ]

        vectors = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        self.faiss_index.add(vectors)

        for item, normalized in zip(entries, texts):
            item["normalized_claim"] = normalized
            item["claim_hash"] = self._claim_hash(normalized)
            self.cache_metadata.append(item)

    # ============================================================
    # Tier 1 exact cache lookup
    # ============================================================

    def _exact_cache_lookup(self, normalized_query):
        """Return an exact previously verified session result."""

        claim_hash = self._claim_hash(normalized_query)

        for item in self.cache_metadata:
            if item.get("claim_hash") == claim_hash:
                return self._format_cached_result(item)

        return None

    # ============================================================
    # Tier 1 semantic candidate
    # ============================================================

    def _semantic_cache_candidate(self, normalized_query):
        """
        Find the closest previously verified claim.

        IMPORTANT:
        This only returns a candidate. It does not automatically trust
        the candidate because semantically similar claims can have
        opposite meanings.
        """

        if self.faiss_index.ntotal == 0:
            return None

        query_vector = self.embedder.encode(
            [normalized_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        distances, indices = self.faiss_index.search(
            query_vector,
            1
        )

        distance = float(distances[0][0])
        index = int(indices[0][0])

        if index == -1:
            return None

        if distance > TIER1_DISTANCE_THRESHOLD:
            return None

        return {
            "distance": distance,
            "metadata": self.cache_metadata[index]
        }

    def _format_cached_result(self, item):
        return {
            "tier": 1,
            "label": item["label"],
            "confidence_score": item.get(
                "confidence_score",
                "Previously verified"
            ),
            "reason": item["reason"],
            "sources": item.get("sources", []),
            "verification_type": item.get(
                "verification_type",
                "previously_verified"
            ),
            "matched_claim": item["text"]
        }

    # ============================================================
    # Live web search
    # ============================================================

    def _fetch_live_context(self, query):
        """
        Fetch live search results and PRESERVE source metadata.

        The LLM never has to invent source names or URLs. It receives
        numbered sources and returns source indexes.
        """

        results = []

        try:
            with DDGS() as ddgs:
                for r in ddgs.text(
                    query,
                    max_results=MAX_SEARCH_RESULTS
                ):
                    title = (r.get("title") or "").strip()
                    body = (r.get("body") or "").strip()
                    url = (r.get("href") or r.get("url") or "").strip()

                    if not title and not body:
                        continue

                    domain = ""
                    if url:
                        try:
                            domain = urlparse(url).netloc.lower()
                            if domain.startswith("www."):
                                domain = domain[4:]
                        except Exception:
                            domain = ""

                    results.append({
                        "index": len(results),
                        "name": title or domain or "Unknown source",
                        "domain": domain,
                        "url": url,
                        "snippet": body
                    })

            if not results:
                return [], "No live search results available."

            context_parts = []

            for source in results:
                context_parts.append(
                    f"SOURCE [{source['index']}]\n"
                    f"Name: {source['name']}\n"
                    f"Domain: {source['domain']}\n"
                    f"URL: {source['url']}\n"
                    f"Snippet: {source['snippet']}"
                )

            return results, "\n\n".join(context_parts)

        except Exception as e:
            print(f"[!] Web search failed: {e}")
            return [], "No live search results available."

    # ============================================================
    # Shared deterministic verification prompt
    # ============================================================

    def _build_messages(
        self,
        query,
        live_context,
        candidate_claim=None,
        candidate_result=None
    ):
        system_prompt = """You are TruthLens, a strict factual verification
engine.

Your task is to classify a user claim using ONLY the supplied evidence.

Possible labels:
- "Real": reliable evidence supports the claim.
- "Fake": reliable evidence clearly contradicts or debunks the claim.
- "Unverified": evidence is insufficient, ambiguous, outdated, or conflicting.

IMPORTANT RULES:

1. Do not use your own memory as evidence.
2. Do not invent sources, URLs, organizations, dates, or quotations.
3. A source is not automatically reliable merely because it appears in
   search results.
4. Prefer authoritative primary sources and established reputable
   organizations.
5. Multiple independent sources are stronger than several copies of the
   same report.
6. If the evidence does not clearly support either side, return
   "Unverified".
7. For a previously verified candidate, check whether the NEW claim has
   exactly the same meaning. Similar wording is NOT sufficient.
8. Be especially careful with negation. For example:
   "X does not happen" and "X happens" must not be treated as the same claim.

Return ONLY valid JSON. No Markdown and no text outside the JSON.

Required JSON format:

{
  "label": "Fake" | "Real" | "Unverified",
  "evidence_strength": 0,
  "source_credibility": 0,
  "llm_certainty": 0,
  "supporting_sources": [0],
  "contradicting_sources": [1],
  "reason": "concise explanation"
}

All scores must be integers from 0 to 100.

supporting_sources and contradicting_sources MUST contain only the
SOURCE indexes supplied in the evidence.
"""

        user_prompt = (
            f"USER CLAIM:\n{query}\n\n"
            f"LIVE WEB EVIDENCE:\n{live_context}"
        )

        if candidate_claim and candidate_result:
            user_prompt += (
                "\n\nPREVIOUSLY VERIFIED SESSION CLAIM:\n"
                f"{candidate_claim}\n\n"
                "PREVIOUSLY VERIFIED RESULT:\n"
                f"{json.dumps(candidate_result, ensure_ascii=False)}\n\n"
                "Determine whether the NEW USER CLAIM has the same factual "
                "meaning as the previously verified claim. Only reuse it "
                "if the meaning is genuinely equivalent."
            )

        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

    # ============================================================
    # LLM response parsing
    # ============================================================

    @staticmethod
    def _extract_json(response_text):
        """
        Robustly extract a JSON object from an LLM response.

        Handles:
        - normal JSON
        - ```json ... ``` responses
        - extra text around JSON
        - nested JSON objects
        - truncated/incomplete JSON
        """

        response_text = (response_text or "").strip()

        if not response_text:
            raise ValueError("Model returned an empty response.")

        # Remove Markdown code fences if the model used them.
        cleaned = re.sub(
            r"^```(?:json)?\\s*",
            "",
            response_text,
            flags=re.IGNORECASE
        )
        cleaned = re.sub(
            r"\\s*```$",
            "",
            cleaned
        ).strip()

        # First try the complete response directly.
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Find a balanced JSON object instead of using greedy
        # regex matching. This correctly handles nested objects.
        start = cleaned.find("{")

        if start == -1:
            raise ValueError(
                f"Model did not return JSON: {response_text}"
            )

        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(cleaned)):
            char = cleaned[i]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                if depth == 0:
                    candidate = cleaned[start:i + 1]

                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Model returned malformed JSON: "
                            f"{candidate}"
                        ) from e

        raise ValueError(
            "Model returned truncated/incomplete JSON: "
            f"{response_text}"
        )

    def _call_and_parse_with_retry(
        self,
        caller,
        messages,
        caller_name
    ):
        """
        Call an LLM and parse its JSON response.

        If the first response is truncated or malformed, retry once
        with a much smaller response schema. This is specifically
        designed to recover from outputs like:

        {
          "label": "Real",
          "evidence_strength": 90,
          "source_credibility":

        The retry asks the model for only the fields that are essential
        for the decision. Python calculates confidence and source
        metadata afterward.
        """

        first_error = None

        # --------------------------------------------------------
        # Attempt 1: normal structured response
        # --------------------------------------------------------
        try:
            raw = caller(messages)

            print(
                f"[*] {caller_name} Raw Output: {raw}"
            )

            return self._extract_json(raw)

        except Exception as e:
            first_error = e

            print(
                f"[!] {caller_name} first response could not be "
                f"parsed: {e}"
            )

        # --------------------------------------------------------
        # Attempt 2: minimal deterministic JSON
        # --------------------------------------------------------
        retry_system = """Return ONLY this minimal JSON object:

{
  "label": "Fake" | "Real" | "Unverified",
  "supporting_sources": [],
  "contradicting_sources": [],
  "reason": "brief explanation"
}

Do not include any other fields.
Do not use Markdown.
Do not add any text outside the JSON.
Use only source indexes that exist in the supplied evidence.
If evidence is insufficient, use "Unverified".
"""

        retry_messages = [
            {
                "role": "system",
                "content": retry_system
            },
            {
                "role": "user",
                "content": messages[-1]["content"]
            }
        ]

        try:
            print(
                f"   -> Retrying {caller_name} with "
                f"minimal JSON schema..."
            )

            raw = caller(retry_messages)

            print(
                f"[*] {caller_name} Retry Raw Output: {raw}"
            )

            data = self._extract_json(raw)

            # The retry does not provide scores, so use conservative
            # deterministic values based on whether it identified
            # supporting/contradicting sources.
            support = data.get(
                "supporting_sources",
                []
            )
            contradict = data.get(
                "contradicting_sources",
                []
            )

            if data.get("label") in {"Real", "Fake"}:
                if support or contradict:
                    data["evidence_strength"] = 85
                    data["source_credibility"] = 75
                    data["llm_certainty"] = 85
                else:
                    data["evidence_strength"] = 55
                    data["source_credibility"] = 50
                    data["llm_certainty"] = 65
            else:
                data["evidence_strength"] = 40
                data["source_credibility"] = 40
                data["llm_certainty"] = 80

            return data

        except Exception as retry_error:
            raise RuntimeError(
                f"{caller_name} failed after JSON retry. "
                f"First error: {first_error}. "
                f"Retry error: {retry_error}"
            ) from retry_error

    @staticmethod
    def _safe_score(value):
        try:
            value = int(float(value))
        except (TypeError, ValueError):
            value = 50

        return max(0, min(100, value))

    @staticmethod
    def _safe_source_indexes(value, source_count):
        if not isinstance(value, list):
            return []

        output = []

        for item in value:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue

            if 0 <= index < source_count and index not in output:
                output.append(index)

        return output

    def _calculate_confidence(
        self,
        evidence,
        credibility,
        certainty
    ):
        weighted_sum = (
            evidence * TIER2_WEIGHTS["evidence_strength"]
            + credibility * TIER2_WEIGHTS["source_credibility"]
            + certainty * TIER2_WEIGHTS["llm_certainty"]
        )

        total_weight = sum(TIER2_WEIGHTS.values())

        return round(
            weighted_sum / total_weight,
            1
        )

    # ============================================================
    # Convert model output into final result
    # ============================================================

    def _build_result(
        self,
        llm_data,
        sources,
        tier=2,
        verification_type="live_web"
    ):
        label = llm_data.get(
            "label",
            "Unverified"
        )

        if label not in {
            "Fake",
            "Real",
            "Unverified"
        }:
            label = "Unverified"

        evidence = self._safe_score(
            llm_data.get("evidence_strength", 50)
        )

        credibility = self._safe_score(
            llm_data.get("source_credibility", 50)
        )

        certainty = self._safe_score(
            llm_data.get("llm_certainty", 50)
        )

        confidence = self._calculate_confidence(
            evidence,
            credibility,
            certainty
        )

        supporting = self._safe_source_indexes(
            llm_data.get("supporting_sources", []),
            len(sources)
        )

        contradicting = self._safe_source_indexes(
            llm_data.get("contradicting_sources", []),
            len(sources)
        )

        selected_sources = []

        for index in supporting:
            source = dict(sources[index])
            source.pop("snippet", None)
            source["role"] = "supporting"
            selected_sources.append(source)

        for index in contradicting:
            if index in supporting:
                continue

            source = dict(sources[index])
            source.pop("snippet", None)
            source["role"] = "contradicting"
            selected_sources.append(source)

        # If the model failed to identify sources but the verdict is
        # Real/Fake, do not invent attribution. Keep the list empty.
        return {
            "tier": tier,
            "label": label,
            "confidence_score": f"{confidence}%",
            "breakdown": {
                "evidence_strength": evidence,
                "source_credibility": credibility,
                "model_certainty": certainty
            },
            "reason": llm_data.get(
                "reason",
                "No verification reason provided."
            ),
            "sources": selected_sources,
            "verification_type": verification_type
        }

    # ============================================================
    # Primary LLM: Hugging Face
    # ============================================================

    def _call_huggingface(self, messages):
        if not self.hf_client:
            raise RuntimeError(
                "Hugging Face primary model is not configured."
            )

        response = self.hf_client.chat_completion(
            model=HF_MODEL_ID,
            messages=messages,
            max_tokens=500,
            temperature=0.0
        )

        return response.choices[0].message.content

    # ============================================================
    # Fallback LLM: Groq Console
    # ============================================================

    def _call_groq(self, messages):
        if not self.groq_client:
            raise RuntimeError(
                "Groq fallback is not configured. "
                "Add GROQ_API_KEY to .env."
            )

        # Groq's API is OpenAI-compatible.
        # temperature=0 reduces generation variability.
        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    # ============================================================
    # Tier 1 semantic verification
    # ============================================================

    def _try_tier1_semantic_match(
        self,
        query,
        normalized_query,
        sources,
        live_context
    ):
        """
        A semantic match is only a CANDIDATE.

        We send the candidate + current evidence to the LLM so that
        opposite claims are not accidentally reused.
        """

        candidate = self._semantic_cache_candidate(
            normalized_query
        )

        if not candidate:
            return None

        cached = candidate["metadata"]

        print(
            f"   -> Tier 1 semantic candidate found "
            f"(distance={candidate['distance']:.4f})"
        )

        cached_result = self._format_cached_result(cached)

        messages = self._build_messages(
            query=query,
            live_context=live_context,
            candidate_claim=cached["text"],
            candidate_result=cached_result
        )

        # Try primary first, then Grok.
        llm_errors = []

        if self.hf_client:
            try:
                raw = self._call_huggingface(messages)
                data = self._extract_json(raw)

                certainty = self._safe_score(
                    data.get("llm_certainty", 0)
                )

                same_label = (
                    data.get("label") == cached["label"]
                )

                if (
                    same_label
                    and certainty >= TIER1_REUSE_CONFIDENCE
                ):
                    result = self._format_cached_result(cached)
                    result["tier"] = 1
                    result["verification_type"] = (
                        "session_cache_semantic_match"
                    )
                    return result

            except Exception as e:
                llm_errors.append(f"Hugging Face: {e}")

        if self.groq_client:
            try:
                raw = self._call_groq(messages)
                data = self._extract_json(raw)

                certainty = self._safe_score(
                    data.get("llm_certainty", 0)
                )

                same_label = (
                    data.get("label") == cached["label"]
                )

                if (
                    same_label
                    and certainty >= TIER1_REUSE_CONFIDENCE
                ):
                    result = self._format_cached_result(cached)
                    result["tier"] = 1
                    result["verification_type"] = (
                        "session_cache_semantic_match"
                    )
                    return result

            except Exception as e:
                llm_errors.append(f"Groq: {e}")

        if llm_errors:
            print(
                "[!] Tier 1 semantic confirmation unavailable. "
                "Falling back to Tier 2."
            )

        return None

    # ============================================================
    # Tier 2
    # ============================================================

    def _tier_2_live_rag(self, query):
        """
        Tier 2:
        1. Fetch live web evidence.
        2. Try Hugging Face primary LLM.
        3. If HF fails, use Grok.
        4. Preserve actual source names/URLs from the web search.
        """

        sources, live_context = self._fetch_live_context(query)

        if not sources:
            print(
                "[!] No live web evidence was found."
            )

        messages = self._build_messages(
            query=query,
            live_context=live_context
        )

        # --------------------------------------------------------
        # Primary: Hugging Face
        # --------------------------------------------------------

        if self.hf_client:
            try:
                print("   -> Trying primary LLM (Hugging Face)...")

                data = self._call_and_parse_with_retry(
                    self._call_huggingface,
                    messages,
                    "Hugging Face"
                )

                return self._build_result(
                    data,
                    sources,
                    tier=2,
                    verification_type="live_web_huggingface"
                )

            except Exception as e:
                print(
                    f"[!] Hugging Face failed: {e}"
                )
                print(
                    "   -> Switching to Groq fallback..."
                )

        # --------------------------------------------------------
        # Fallback: Grok Console
        # --------------------------------------------------------

        if self.groq_client:
            try:
                print("   -> Trying fallback LLM (Groq)...")

                data = self._call_and_parse_with_retry(
                    self._call_groq,
                    messages,
                    "Groq"
                )

                return self._build_result(
                    data,
                    sources,
                    tier=2,
                    verification_type="live_web_groq_fallback"
                )

            except Exception as e:
                print(
                    f"[!] Groq fallback failed: {e}"
                )

                return {
                    "tier": 2,
                    "label": "Unverified",
                    "confidence_score": "0%",
                    "breakdown": {
                        "evidence_strength": 0,
                        "source_credibility": 0,
                        "model_certainty": 0
                    },
                    "reason": (
                        "Both the primary and fallback LLM failed. "
                        "The claim cannot be safely verified."
                    ),
                    "sources": [
                        {
                            "name": source["name"],
                            "domain": source["domain"],
                            "url": source["url"],
                            "role": "retrieved evidence"
                        }
                        for source in sources
                    ],
                    "verification_type": "verification_failed"
                }

        return {
            "tier": 2,
            "label": "Unverified",
            "confidence_score": "0%",
            "breakdown": {
                "evidence_strength": 0,
                "source_credibility": 0,
                "model_certainty": 0
            },
            "reason": (
                "No usable LLM provider is configured."
            ),
            "sources": [
                {
                    "name": source["name"],
                    "domain": source["domain"],
                    "url": source["url"],
                    "role": "retrieved evidence"
                }
                for source in sources
            ],
            "verification_type": "no_llm_available"
        }

    # ============================================================
    # Store high-confidence Tier 2 result in session cache
    # ============================================================

    def _store_verified_result(
        self,
        query,
        result
    ):
        """
        Add a high-confidence result to the in-memory Tier 1 cache.

        It is intentionally NOT persisted to disk.
        """

        if result.get("label") not in {"Real", "Fake"}:
            return

        try:
            confidence = float(
                str(
                    result.get(
                        "confidence_score",
                        "0"
                    )
                ).replace("%", "")
            )
        except ValueError:
            confidence = 0

        breakdown = result.get("breakdown", {})

        evidence = float(
            breakdown.get("evidence_strength", 0)
        )

        credibility = float(
            breakdown.get("source_credibility", 0)
        )

        certainty = float(
            breakdown.get("model_certainty", 0)
        )

        # Only cache strong results.
        if confidence < 85:
            return

        if evidence < 80:
            return

        if credibility < 70:
            return

        if certainty < 80:
            return

        normalized = self._normalize_claim(query)

        # Don't duplicate an exact claim.
        if self._exact_cache_lookup(normalized):
            return

        cache_item = {
            "text": query,
            "label": result["label"],
            "confidence_score": result["confidence_score"],
            "reason": result["reason"],
            "sources": result.get("sources", []),
            "verification_type": (
                "session_verified_tier2"
            )
        }

        self._add_cache_entries([cache_item])

        print(
            "   -> High-confidence result stored in "
            "Tier 1 session cache."
        )

    # ============================================================
    # Main verification
    # ============================================================

    def verify_claim(self, query):
        query = query.strip()

        if not query:
            return {
                "tier": 0,
                "label": "Error",
                "reason": "Claim cannot be empty."
            }

        normalized_query = self._normalize_claim(query)

        print(f"\n[+] Analyzing: '{query}'")

        # --------------------------------------------------------
        # Tier 1A: exact session match
        # --------------------------------------------------------

        exact = self._exact_cache_lookup(
            normalized_query
        )

        if exact:
            print(
                "   -> Tier 1 EXACT CACHE HIT "
                "(session memory)"
            )
            return exact

        # --------------------------------------------------------
        # Search evidence once for this new claim.
        # This evidence is then used for semantic cache checking
        # AND Tier 2, preventing different evidence between models.
        # --------------------------------------------------------

        sources, live_context = self._fetch_live_context(
            query
        )

        # --------------------------------------------------------
        # Tier 1B: semantic candidate
        # --------------------------------------------------------

        semantic_result = self._try_tier1_semantic_match(
            query=query,
            normalized_query=normalized_query,
            sources=sources,
            live_context=live_context
        )

        if semantic_result:
            print(
                "   -> Tier 1 SEMANTIC CACHE HIT "
                "(confirmed)"
            )
            return semantic_result

        # --------------------------------------------------------
        # Tier 2
        # --------------------------------------------------------

        print(
            "   -> Tier 1 MISS. "
            "Triggering Tier 2 Live RAG..."
        )

        # Reuse the SAME search evidence that was already fetched.
        # This avoids searching twice.
        result = self._tier_2_with_existing_evidence(
            query,
            sources,
            live_context
        )

        # --------------------------------------------------------
        # Store strong results in session cache
        # --------------------------------------------------------

        self._store_verified_result(
            query,
            result
        )

        return result

    def _tier_2_with_existing_evidence(
        self,
        query,
        sources,
        live_context
    ):
        messages = self._build_messages(
            query=query,
            live_context=live_context
        )

        # Primary LLM
        if self.hf_client:
            try:
                print(
                    "   -> Trying primary LLM (Hugging Face)..."
                )

                data = self._call_and_parse_with_retry(
                    self._call_huggingface,
                    messages,
                    "Hugging Face"
                )

                return self._build_result(
                    data,
                    sources,
                    tier=2,
                    verification_type="live_web_huggingface"
                )

            except Exception as e:
                print(
                    f"[!] Hugging Face failed: {e}"
                )
                print(
                    "   -> Switching to Groq fallback..."
                )

        # Groq fallback
        if self.groq_client:
            try:
                print(
                    "   -> Trying fallback LLM (Groq)..."
                )

                data = self._call_and_parse_with_retry(
                    self._call_groq,
                    messages,
                    "Groq"
                )

                return self._build_result(
                    data,
                    sources,
                    tier=2,
                    verification_type="live_web_groq_fallback"
                )

            except Exception as e:
                print(
                    f"[!] Groq fallback failed: {e}"
                )

        return {
            "tier": 2,
            "label": "Unverified",
            "confidence_score": "0%",
            "breakdown": {
                "evidence_strength": 0,
                "source_credibility": 0,
                "model_certainty": 0
            },
            "reason": (
                "The claim could not be safely verified because "
                "the available LLM providers failed."
            ),
            "sources": [
                {
                    "name": source["name"],
                    "domain": source["domain"],
                    "url": source["url"],
                    "role": "retrieved evidence"
                }
                for source in sources
            ],
            "verification_type": "verification_failed"
        }


# ============================================================
# Run Script
# ============================================================

if __name__ == "__main__":
    try:
        engine = TextVerificationPipeline()

        while True:
            user_input = input(
                "\nEnter a claim to verify "
                "(or type 'quit' to exit): "
            )

            if user_input.lower().strip() in {
                "quit",
                "exit"
            }:
                print("Exiting...")
                break

            result = engine.verify_claim(
                user_input
            )

            print("\nRESULT:")
            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False
                )
            )

    except KeyboardInterrupt:
        print("\nExiting...")

    except Exception as e:
        print(
            f"\n[FATAL ERROR] {e}"
        )
