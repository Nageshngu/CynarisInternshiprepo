"""Integration checks against real model via FastAPI TestClient."""
import unittest
from fastapi.testclient import TestClient
from backend.main import app

VALID = {"Age": 35, "Gender": "F", "Fever": 1, "Cough": 1,
         "Fatigue": 0, "Breathlessness": 0, "Chest_Pain": 0,
         "Nausea": 0, "Headache": 0, "BMI": 24.0,
         "BP_Systolic": 120, "BP_Diastolic": 80, "SpO2": 98,
         "Glucose_mg_dL": 95, "HbA1c": 5.2}

class TestWeek9Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["model_available"])

    def test_prediction(self):
        r = self.client.post("/predict", json=VALID)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("predicted_condition", body)
        self.assertGreater(len(body["model_scores"]), 0)
        self.assertIn("not a diagnosis", body["disclaimer"].lower())

    def test_missing_feature_rejected(self):
        bad = dict(VALID)
        del bad["Age"]
        self.assertEqual(self.client.post("/predict", json=bad).status_code, 422)

    def test_invalid_value_rejected(self):
        self.assertEqual(self.client.post("/predict", json={**VALID, "SpO2": 110}).status_code, 422)

    def test_warning_signs(self):
        r = self.client.post("/predict", json={**VALID, "Chest_Pain": 1})
        self.assertEqual(r.status_code, 200)
        self.assertIn("warning signs", r.json()["safety_message"])

if __name__ == "__main__":
    unittest.main()
