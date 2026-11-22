"""Week 1 API skeleton: no diagnostic predictions available."""
from fastapi import FastAPI

app = FastAPI(title='Educational Symptom Checker — Week 1')

@app.get('/health')
def health():
    return {'status': 'ok', 'phase': 'week1_architecture_only'}
