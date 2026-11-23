"""50-concurrent-request benchmark. Run with API already running.
Usage: python scripts/load_test_week9.py --concurrency 50 --requests 200
Results are observed measurements; no benchmark values are prefilled.
"""
import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sample_inputs():
    df = pd.read_csv(ROOT / "data/raw/healthcare_symptoms.csv")
    return df.drop(columns=["Patient_ID", "Diagnosis"]).to_dict(orient="records")

async def benchmark(url, concurrency, total):
    samples = sample_inputs()
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    async with httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=concurrency)) as client:
        async def one(i):
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(url, json=samples[i % len(samples)])
                    return {"status": response.status_code, "latency_ms": (time.perf_counter()-started)*1000}
                except httpx.RequestError:
                    return {"status": 0, "latency_ms": (time.perf_counter()-started)*1000}
        start = time.perf_counter()
        results = await asyncio.gather(*(one(i) for i in range(total)))
        elapsed = time.perf_counter()-start
    latencies = sorted(r["latency_ms"] for r in results)
    def percentile(p):
        return round(latencies[min(len(latencies)-1, int((len(latencies)-1)*p))], 2)
    successes = sum(r["status"] == 200 for r in results)
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "url": url,
            "concurrency": concurrency, "requests": total, "successful": successes,
            "failed": total-successes, "elapsed_seconds": round(elapsed, 3),
            "throughput_requests_per_second": round(total/elapsed, 2),
            "mean_latency_ms": round(statistics.mean(latencies), 2),
            "p50_latency_ms": percentile(.50), "p95_latency_ms": percentile(.95),
            "p99_latency_ms": percentile(.99),
            "status_counts": {str(s): sum(r["status"] == s for r in results) for s in sorted(set(r["status"] for r in results))}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must be positive")
    result = asyncio.run(benchmark(args.url, args.concurrency, args.requests))
    out = ROOT / "reports/week9_load_test.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved observed results to {out}")
