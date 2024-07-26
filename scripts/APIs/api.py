import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from datetime import datetime

# Load the preprocessor and model from the pickle files
with open('./models/preprocessed.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

with open('./models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define Pydantic model for request
class HealthData(BaseModel):
    Timestamp: str
    Age: int
    Gender: str
    Country: str
    state: Optional[str]
    self_employed: Optional[bool]
    family_history: Optional[bool]
    treatment: str
    work_interfere: Optional[str]
    no_employees: str
    remote_work: Optional[bool]
    tech_company: Optional[bool]
    benefits: Optional[str]
    care_options: Optional[str]
    wellness_program: Optional[str]
    seek_help: Optional[str]
    anonymity: Optional[str]
    leave: Optional[str]
    mental_health_consequence: Optional[str]
    phys_health_consequence: Optional[str]
    coworkers: Optional[str]
    supervisor: Optional[str]
    mental_health_interview: Optional[str]
    phys_health_interview: Optional[str]
    mental_vs_physical: Optional[str]
    obs_consequence: Optional[str]

@app.post("/predict")
def predict(data: HealthData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Preprocess the new data
    input_data_transformed = preprocessor.transform(input_data.drop(columns=['treatment'], axis=1))

    # Make a prediction
    prediction = model.predict(input_data_transformed)[0]

    # Map prediction to human-readable output
    result = 'Yes' if prediction == 1 else 'No'

    return {"prediction": result}

# Endpoint to check the health of the API
@app.get("/")
def read_root():
    return {"message": "Health Tracker API is up and running!"}
