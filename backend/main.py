"""Week 9 educational model-serving API. Not for clinical use."""
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "week6_best_random_forest.joblib"
FEATURES = ["Age", "Gender", "Fever", "Cough", "Fatigue", "Breathlessness",
            "Chest_Pain", "Nausea", "Headache", "BMI", "BP_Systolic",
            "BP_Diastolic", "SpO2", "Glucose_mg_dL", "HbA1c"]

class Symptoms(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Age: int = Field(ge=0, le=120)
    Gender: str
    Fever: int = Field(ge=0, le=1)
    Cough: int = Field(ge=0, le=1)
    Fatigue: int = Field(ge=0, le=1)
    Breathlessness: int = Field(ge=0, le=1)
    Chest_Pain: int = Field(ge=0, le=1)
    Nausea: int = Field(ge=0, le=1)
    Headache: int = Field(ge=0, le=1)
    BMI: float = Field(gt=0, le=100)
    BP_Systolic: float = Field(gt=0, le=300)
    BP_Diastolic: float = Field(gt=0, le=200)
    SpO2: float = Field(ge=0, le=100)
    Glucose_mg_dL: float = Field(gt=0, le=1000)
    HbA1c: float = Field(gt=0, le=25)

@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train Week 6 first.")
    # Load only project-owned, trusted joblib artifacts (pickle-based format).
    return joblib.load(MODEL_PATH)

app = FastAPI(title="HealthBridge Educational Symptom Checker", version="0.9.0")

@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.is_file(), "disclaimer": "Educational demonstration only"}

@app.post("/predict")
def predict(payload: Symptoms):
    values = payload.model_dump()
    if values["Gender"] not in {"M", "F"}:
        raise HTTPException(status_code=422, detail="Gender must be M or F for this training dataset")
    if values["Chest_Pain"] or values["Breathlessness"] or values["SpO2"] < 94:
        safety = "Potential warning signs reported. Seek prompt medical assessment; severe symptoms require emergency care."
    else:
        safety = "Seek professional medical advice if symptoms persist, worsen, or concern you."
    try:
        model = get_model()
        frame = pd.DataFrame([{k: values[k] for k in FEATURES}], columns=FEATURES)
        label = str(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        classes = model.classes_
        scores = sorted(({"condition": str(c), "model_score": round(float(s), 4)} for c, s in zip(classes, probabilities)), key=lambda x: x["model_score"], reverse=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"predicted_condition": label, "model_scores": scores[:3],
            "safety_message": safety,
            "disclaimer": "Research demonstration, not a diagnosis. Model scores are not validated clinical probabilities."}
