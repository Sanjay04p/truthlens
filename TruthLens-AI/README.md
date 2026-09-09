# TruthLens-AI

An AI-powered misinformation detection system that analyzes **text claims and images** to determine whether content is **Real, Fake, or Unverified**.

The system combines **semantic similarity search, FAISS caching, Large Language Models (LLMs), web-based fact verification, and AI-generated image detection** to provide fast and reliable results.

---

## 🚀 Features

* 📰 **Text Claim Verification**
* 🔍 **Semantic Matching using FAISS**
* 🌐 **Live Web Verification using DuckDuckGo**
* 🤖 **LLM-based Fact Classification**
* 🖼️ **AI-Generated Image Detection**
* 🔄 **Fallback Image Detection using Sightengine**
* ⚡ **In-Memory Caching**
* 🎯 **Confidence-Based Result Caching**
* ⚠️ **Negation and Contradiction Detection**
* 📊 **Evidence Strength & Source Credibility Analysis**
* 🧾 **Structured JSON Responses**
* 🧹 **Automatic Temporary File Cleanup**

---

## 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │   User Input     │
                    │ Text / Image     │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
             Text Pipeline          Image Pipeline
                 │                       │
                 ▼                       ▼
          Exact Match Check       Hugging Face Model
                 │                       │
                 ▼                  Success?
          Generate Embedding             │
                 │                  ┌────┴────┐
                 ▼                  │         │
              FAISS Search         Yes       No
                 │                  │         │
                 ▼                  │    Sightengine
          Similarity Threshold     │         │
                 │                  └────┬────┘
                 ▼                       │
          LLM Meaning Check              ▼
                 │                 AI Probability
        ┌────────┴────────┐               │
        │                 │               ▼
   Equivalent?         Not Equivalent   LLM Analysis
        │                 │               │
        ▼                 ▼               ▼
  Cached Result      Live Verification  Final Result
                          │
                          ▼
                    DuckDuckGo Search
                          │
                          ▼
                     Up to 5 Sources
                          │
                          ▼
                    LLM Classification
                          │
                          ▼
                Real / Fake / Unverified
                          │
                          ▼
                 Confidence Calculation
                          │
                          ▼
                    Cache if Eligible
```

---

# 📰 Text Verification Pipeline

The text pipeline uses a **tiered verification strategy** to reduce unnecessary web searches and improve response speed.

## Tier 1: Exact Match

When a claim is received:

1. The system checks the in-memory cache for an exact match.
2. If a matching claim is found, the cached result is returned.
3. If no exact match exists, semantic matching is performed.

---

## Tier 1: Semantic Matching

If an exact match is unavailable:

1. The claim is converted into an **embedding/vector representation**.
2. **FAISS** searches for semantically similar previously verified claims.
3. A similarity threshold determines whether a suitable candidate exists.
4. The LLM verifies whether the new claim has the **same meaning** as the retrieved claim.
5. The system specifically checks for:

   * Negation
   * Contradictions
   * Opposite meanings
   * Differences in context

A previous result is reused only when:

* The meaning is genuinely equivalent.
* The predicted labels match.
* Model certainty is at least **85%**.

If these conditions are not satisfied, the system proceeds to live verification.

---

# 🌐 Tier 2: Live Web Verification

When a reliable cached result cannot be reused, the system performs live verification.

### Process

1. Search the web using **DuckDuckGo Search**.
2. Collect up to **5 relevant sources**.
3. Extract:

   * Source title
   * Domain
   * URL
   * Search snippet
4. Provide the collected evidence to the LLM using a strict JSON prompt.
5. The LLM classifies the claim as:

```text
Real
Fake
Unverified
```

The LLM also returns:

* Evidence strength
* Source credibility
* Model certainty
* Supporting source indexes
* Contradicting source indexes
* Reason for the verdict

---

# 🎯 Confidence & Caching

The backend calculates a final confidence score using:

```text
Evidence Strength
        +
Source Credibility
        +
Model Certainty
        ↓
Final Confidence
```

A strong Tier 2 result is added to the in-memory FAISS cache only when:

| Condition          |   Threshold |
| ------------------ | ----------: |
| Verdict            | Real / Fake |
| Final Confidence   |       ≥ 85% |
| Evidence Strength  |       ≥ 80% |
| Source Credibility |       ≥ 70% |
| Model Certainty    |       ≥ 80% |

This prevents weak or uncertain results from being reused in future requests.

---

# 🖼️ Image Analysis Pipeline

The image pipeline determines whether an uploaded image is likely to be **AI-generated or real**.

## Step 1: Hugging Face Detection

The system first sends the image to the Hugging Face model:

```text
dima806/ai_vs_real_image_detection
```

The model calculates the probability that the image is AI-generated.

---

## Step 2: Fallback Detection

If the Hugging Face service fails, the system automatically uses **Sightengine** with its GenAI detection model.

```text
Hugging Face
     │
     ├── Success → AI Probability
     │
     └── Failure → Sightengine → AI Probability
```

This fallback mechanism improves reliability when the primary detection service is unavailable.

---

# 🤖 LLM Processing

The LLM is used to interpret verification results and produce a structured response.

The system uses a **strict JSON prompt** to ensure predictable output.

If the first LLM response contains invalid JSON:

1. The response is detected as invalid.
2. The request is retried.
3. A smaller and simpler JSON structure is requested.
4. The corrected response is parsed.

This makes the pipeline more robust against malformed LLM responses.

---

# 🗂️ Temporary File Management

Uploaded images are processed using temporary files.

The temporary image file is deleted inside the Python `finally` block after processing.

This ensures that temporary files are cleaned up even when an exception occurs.

```text
Upload Image
     ↓
Temporary File
     ↓
Image Analysis
     ↓
LLM Processing
     ↓
Delete Temporary File
```

---

# ⚡ Caching Strategy

The project uses an **in-memory FAISS cache** for previously verified claims.

### Advantages

* Faster responses for repeated claims
* Reduces unnecessary web searches
* Reduces repeated LLM calls
* Improves overall system efficiency

### Cache Limitation

The cache exists only for the **current backend process**.

It is:

* Not stored in a database
* Not persistent across server restarts
* Cleared when the backend process terminates

---

# 🔄 Overall Workflow

```text
                  USER INPUT
                      │
             ┌────────┴────────┐
             │                 │
           TEXT              IMAGE
             │                 │
             ▼                 ▼
       Exact Match       Hugging Face
             │                 │
        Match Found?       Successful?
        /          \        /       \
      Yes           No    Yes        No
       │             │     │          │
       ▼             ▼     ▼          ▼
   Cached       Embedding  AI       Sightengine
   Result          │      Score        │
                   ▼        └────┬─────┘
                 FAISS          │
                   │            ▼
                   ▼       LLM Analysis
             LLM Meaning          │
                   │              ▼
             Equivalent?      Final Result
              /       \
            Yes        No
             │          │
             ▼          ▼
          Cached     DuckDuckGo
          Result       Search
                         │
                         ▼
                    Up to 5 Sources
                         │
                         ▼
                    LLM Verification
                         │
                         ▼
              Real / Fake / Unverified
                         │
                         ▼
                 Confidence Score
                         │
                    ┌────┴────┐
                    │         │
                Strong      Weak
                    │         │
                    ▼         ▼
                 Cache     Don't Cache
```

---

# 🛠️ Technology Stack

| Technology            | Purpose                                       |
| --------------------- | --------------------------------------------- |
| **Python**            | Backend and pipeline implementation           |
| **FAISS**             | Semantic similarity search and vector caching |
| **LLM**               | Claim verification and reasoning              |
| **DuckDuckGo Search** | Live web evidence retrieval                   |
| **Hugging Face**      | AI-generated image detection                  |
| **Sightengine**       | Fallback AI-image detection                   |
| **JSON**              | Structured LLM communication                  |
| **In-Memory Cache**   | Fast reuse of verified results                |

---

# 📁 Suggested Project Structure

```text
project/
│
├── backend/
│   ├── app.py
│   ├── text_pipeline.py
│   ├── image_pipeline.py
│   ├── cache.py
│   ├── search.py
│   └── llm.py
│
├── frontend/
│   └── ...
│
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <project-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

Example:

```env
LLM_API_KEY=your_api_key
SIGHTENGINE_API_USER=your_user
SIGHTENGINE_API_SECRET=your_secret
```

Add any additional API keys required by your implementation.

---

# ▶️ Running the Project

Start the backend using:

```bash
python app.py
```

If your project uses a framework such as FastAPI or Flask, use the corresponding startup command defined in your project.

---

# 🔐 Security

Do not commit API keys or secrets to GitHub.

Add your environment file to `.gitignore`:

```text
.env
venv/
__pycache__/
*.pyc
```

Use `.env.example` to document the required environment variables without exposing actual credentials.

---

# 📊 Output

For text claims, the system provides:

```json
{
  "verdict": "Real",
  "confidence": 91,
  "evidence_strength": 88,
  "source_credibility": 92,
  "model_certainty": 94,
  "supporting_sources": [1, 2],
  "contradicting_sources": [],
  "reason": "The available evidence supports the claim."
}
```

For image analysis, the system returns the AI-generation probability along with the final structured analysis.

---

# 🎯 Key Design Principles

### 1. Fast First

Cached results are checked before expensive live verification.

### 2. Semantic Understanding

FAISS and LLM-based verification allow the system to recognize claims with similar meanings rather than relying only on exact text matching.

### 3. Evidence-Based Verification

Uncached claims are verified using multiple web sources.

### 4. Conservative Caching

Only high-confidence results are stored for future reuse.

### 5. Fault Tolerance

Fallback services and JSON retry mechanisms improve system reliability.

### 6. Temporary Data Cleanup

Temporary image files are removed after processing.

---

# 🚧 Limitations

* Web search results depend on currently available online sources.
* LLM predictions may occasionally be incorrect.
* AI-image detection models are probabilistic and may produce false positives or false negatives.
* The FAISS cache is not persistent and is cleared when the backend restarts.
* An "Unverified" result does not necessarily mean that a claim is false; it means sufficient reliable evidence was not found.

---

# 🔮 Future Improvements

* Persistent vector database for long-term caching
* More advanced source credibility scoring
* Multiple independent LLM verification
* Multilingual claim verification
* Video and audio misinformation detection
* Browser extension for real-time fact checking
* Continuous source monitoring
* Improved explainability and evidence visualization
* User feedback-based model improvement

---

# 👨‍💻 Contributors

**Team Project**

Developed as an AI-based misinformation and media authenticity detection system.

---

# 📜 License

Add your preferred license here, such as **MIT License**, if applicable.

---

## ⭐ Project Summary

This project provides a **multi-layered misinformation detection pipeline** that combines semantic search, FAISS caching, live web verification, LLM reasoning, and AI-generated image detection.

The system is designed to prioritize **speed, reliability, evidence-based decisions, and conservative caching**, while providing users with an interpretable verdict and supporting evidence.
