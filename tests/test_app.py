import unittest
import pickle
import pandas as pd
from sklearn.exceptions import NotFittedError

class TestMentalHealthModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the preprocessor and model from the pickle files
        try:
            with open('preprocesor.pkl', 'rb') as file:
                cls.preprocessor = pickle.load(file)
            print("Preprocessor loaded successfully.")
        except FileNotFoundError:
            print("Error: preprocesor.pkl file not found.")
            exit(1)
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
            exit(1)

        try:
            with open('model.pkl', 'rb') as file:
                cls.model = pickle.load(file)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Error: model.pkl file not found.")
            exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

    def test_predict(self):
        # Create a DataFrame with sample data
        sample_data = pd.DataFrame({
            "Timestamp": ["2014-08-27 11:29:31", "2014-08-27 11:29:37", "2014-08-27 11:29:44", "2014-08-27 11:29:46", "2014-08-27 11:30:22"],
            "Age": [37, 44, 32, 31, 31],
            "Gender": ["Female", "M", "Male", "Male", "Male"],
            "Country": ["United States", "United States", "Canada", "United Kingdom", "United States"],
            "state": ["IL", "IN", None, None, "TX"],
            "self_employed": [None, None, None, None, None],
            "family_history": ["No", "No", "No", "Yes", "No"],
            "treatment": ["Yes", "Yes", "Yes", "Yes", "No"],
            "work_interfere": ["Often", "Rarely", "Rarely", "Often", "Never"],
            "no_employees": ["6-25", "More than 1000", "6-25", "26-100", "100-500"],
            "remote_work": ["No", "No", "No", "No", "Yes"],
            "tech_company": ["Yes", "No", "Yes", "Yes", "Yes"],
            "benefits": ["Yes", "Don't know", "No", "No", "Yes"],
            "care_options": ["Not sure", "No", "No", "Yes", "No"],
            "wellness_program": ["No", "Don't know", "No", "No", "Don't know"],
            "seek_help": ["Yes", "Don't know", "No", "No", "Don't know"],
            "anonymity": ["Yes", "Don't know", "Don't know", "No", "Don't know"],
            "leave": ["Somewhat easy", "Don't know", "Somewhat difficult", "Somewhat difficult", "Don't know"],
            "mental_health_consequence": ["No", "Maybe", "No", "Yes", "No"],
            "phys_health_consequence": ["No", "No", "No", "Yes", "No"],
            "coworkers": ["Some of them", "No", "Yes", "Some of them", "Some of them"],
            "supervisor": ["Yes", "No", "Yes", "No", "Yes"],
            "mental_health_interview": ["No", "No", "Yes", "Maybe", "Yes"],
            "phys_health_interview": ["Maybe", "No", "Yes", "Maybe", "Yes"],
            "mental_vs_physical": ["Yes", "Don't know", "No", "No", "Don't know"],
            "obs_consequence": ["No", "No", "No", "Yes", "No"],
            "comments": [None, None, None, None, None]
        })

        try:
            # Drop the 'treatment' column and preprocess the new data
            new_data_transformed = self.preprocessor.transform(sample_data.drop(columns=['treatment'], axis=1))
            print("Data preprocessed successfully.")

            # Make predictions
            predictions = self.model.predict(new_data_transformed)
            for i, prediction in enumerate(predictions):
                print(f"Prediction for sample {i + 1}: {prediction} (Expected: {sample_data['treatment'][i]})")

            # Assert that the predictions match the expected outcomes
            for i, prediction in enumerate(predictions):
                expected = 1 if sample_data['treatment'][i] == 'Yes' else 0
                self.assertEqual(prediction, expected, f"Expected prediction '{expected}', but got '{prediction}'")

        except NotFittedError as e:
            self.fail(f"Model is not fitted: {e}")
        except Exception as e:
            self.fail(f"An error occurred during prediction: {e}")


# Run the tests
if __name__ == "__main__":
    unittest.main()
