import json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Week6EvidenceTests(unittest.TestCase):
    def test_experiments_present(self):
        data=json.loads((ROOT/'reports/week6_experiments.json').read_text())
        self.assertGreaterEqual(len(data['runs']),2)
        self.assertIn(data['best_run'],[r['run'] for r in data['runs']])
        self.assertEqual(data['split'],{'train':560,'validation':120,'test':120})
    def test_best_model_saved(self):
        self.assertTrue((ROOT/'models/week6_best_random_forest.joblib').exists())
if __name__=='__main__':unittest.main()
