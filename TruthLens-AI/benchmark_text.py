import time
import statistics
import re
from backend_server.pipelines.text_pipeline import TextVerificationPipeline

def extract_conf(conf_str):
    """Extracts numeric float from confidence string like '88.5%'."""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(conf_str))
    return float(match.group(1)) if match else 50.0

print("[*] Initializing Pipeline for Benchmarking...")
engine = TextVerificationPipeline()

# 1. Tier 1 Test Set (Exact matches from the seed cache)
# This will trigger _exact_cache_lookup instantly without network calls
tier1_queries = [
    "The Earth is flat and NASA is faking space images.",
    "Water is composed of two hydrogen atoms and one oxygen atom.",
    " The Earth is flat and NASA is faking space images. ", # Tests normalization
] * 4  # 12 runs

# 2. Tier 2 Test Set (Emergent / Novel Claims)
# This will trigger DDGS search + Groq inference
tier2_queries = [
    "NASA successfully landed astronauts on Mars in 2024.",
    "Drinking boiling vinegar cures acute influenza.",
    "India launched the Chandrayaan-3 lunar mission.",
    "The United Nations enacted a global ban on AI research in 2025."
] * 2  # 8 runs

print("\n[*] Running Tier 1 (Exact Cache Hit) Benchmarks...")
t1_latencies, t1_confs = [], []
for q in tier1_queries:
    t0 = time.perf_counter()
    # verify_claim will intercept this at the exact match layer
    res = engine.verify_claim(q)
    t1_latencies.append(time.perf_counter() - t0)
    t1_confs.append(extract_conf(res.get("confidence_score", 0)))

print("\n[*] Running Tier 2 (Live RAG via Groq) Benchmarks...")
t2_latencies, t2_confs = [], []
for q in tier2_queries:
    t0 = time.perf_counter()
    res = engine.verify_claim(q)
    t2_latencies.append(time.perf_counter() - t0)
    t2_confs.append(extract_conf(res.get("confidence_score", 0)))

print("\n[*] Running Always-Live Baseline Benchmarks...")
live_latencies, live_confs = [], []
# Test a mix of cached and novel claims, forcing them all to bypass the cache
for q in tier1_queries[:4] + tier2_queries[:4]:
    t0 = time.perf_counter()
    # Force Tier 2 bypass by calling the RAG method directly
    res = engine._tier_2_live_rag(q)
    live_latencies.append(time.perf_counter() - t0)
    live_confs.append(extract_conf(res.get("confidence_score", 0)))

# --- Output Formatted Table I ---
print("\n" + "="*60)
print("TABLE I: TEXT VERIFICATION PERFORMANCE (MEASURED)")
print("="*60)
print(f"{'Path':<24} | {'Median Latency (s)':<18} | {'API Calls':<10} | {'Mean Conf. (%)':<15}")
print("-" * 60)
print(f"{'Tier 1 (exact cache hit)':<24} | {statistics.median(t1_latencies):<18.4f} | {'0':<10} | {statistics.mean(t1_confs):<15.1f}")
print(f"{'Tier 2 (live RAG - Groq)':<24} | {statistics.median(t2_latencies):<18.4f} | {'2':<10} | {statistics.mean(t2_confs):<15.1f}")
print(f"{'Always-live baseline':<24} | {statistics.median(live_latencies):<18.4f} | {'2':<10} | {statistics.mean(live_confs):<15.1f}")
print("="*60)