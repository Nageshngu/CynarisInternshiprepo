import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_pipeline import load_dataset,split_data,make_preprocessor

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset(ROOT/'data/raw/healthcare_symptoms.csv')
    def test_split_sizes_and_no_overlap(self):
        (a,b,c),(ya,yb,yc)=split_data(self.df)
        self.assertEqual((len(a),len(b),len(c)),(560,120,120))
        self.assertFalse(set(a.index)&set(b.index))
        self.assertFalse(set(a.index)&set(c.index))
        self.assertFalse(set(b.index)&set(c.index))
    def test_pipeline_transforms(self):
        (a,b,c),_=split_data(self.df)
        p=make_preprocessor(a)
        self.assertEqual(p.fit_transform(a).shape[0],560)
        self.assertEqual(p.transform(b).shape[0],120)
    def test_no_patient_id_as_feature(self):
        (a,_,_),_=split_data(self.df)
        self.assertNotIn('Patient_ID',a.columns)
        self.assertNotIn('Diagnosis',a.columns)
if __name__=='__main__': unittest.main()
