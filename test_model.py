import unittest
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load the preprocessor and model from the pickle files
with open('preprocesor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

class TestModel(unittest.TestCase):

    def test_predict(self):
        # Create a new data point
        new_data = pd.DataFrame({
            "Timestamp": ["2023-10-10 12:00:00"],
            "Age": [35],
            "Gender": ["Male"],
            "Country": ["United States"],
            "state": ["CA"],
            "self_employed": [False],
            "family_history": [True],
            "treatment": ["Yes"],
            "work_interfere": ["Sometimes"],
            "no_employees": [26-100],
            "remote_work": [True],
            "tech_company": [True],
            "benefits": [True],
            "care_options": [True],
            "wellness_program": [True],
            "seek_help": [True],
            "anonymity": [True],
            "leave": [True],
            "mental_health_consequence": [True],
            "phys_health_consequence": [True],
            "coworkers": ["Yes"],
            "supervisor": ["Yes"],
            "mental_health_interview": ["Yes"],
            "phys_health_interview": ["Yes"],
            "mental_vs_physical": ["No"],
            "obs_consequence": ["No"]
        })

        # Preprocess the new data
        new_data_transformed = preprocessor.transform(new_data.drop(columns=['treatment'],axis=1))

        # Make a prediction
        prediction = model.predict(new_data_transformed)[0]

        # Assert that the prediction is 1 (for treatment)
        self.assertEqual(prediction, 1)

