# Week 9 — Running and validating integration

## 1. Setup (Windows PowerShell, from project root)

```powershell
python -m pip install -r requirements.txt
```

## 2. Start backend (Terminal 1)

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Swagger UI: http://127.0.0.1:8000/docs

## 3. Start frontend (Terminal 2)

```powershell
python -m streamlit run frontend/app.py
```

UI: http://127.0.0.1:8501

## 4. Run integration tests (Terminal 3)

```powershell
python -m pytest tests/test_week9_integration.py -v
```

## 5. Load test (Terminal 3, API must be running)

```powershell
python scripts/load_test_week9.py --concurrency 50 --requests 200
```

Actual measurements are written to `reports/week9_load_test.json`. The script reports mean, p50, p95, p99 latency and completed request throughput. Capture the terminal output and MLflow screenshots as evidence. For comparable benchmarks, record the computer and Python versions, server worker count, and whether Streamlit was running. The load test is not a clinical validation study.

## Safety and data notes

- The UI and API are educational demonstrations, not clinical decision support.
- Never send identifiable patient information in this demonstration.
- The model expects the exact 15 training features; Patient_ID and Diagnosis are not inputs.
- Do not load untrusted `.joblib` files; joblib uses pickle serialization.
- Existing model artifact must be available at `models/week6_best_random_forest.joblib`.
