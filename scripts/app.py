import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load the preprocessor and model from the pickle files
with open('preprocesor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the app
def run():
    st.title("Health Tracker Model Testing App")

    # Create inputs for all features
    Timestamp = st.date_input("Timestamp")
    Age = st.number_input("Age", min_value=0, max_value=100)
    Gender = st.selectbox("Gender", ["Male", "Female", "M"])
    Country = st.text_input("Country")
    state = st.text_input("State")
    self_employed = st.checkbox("Self Employed")
    family_history = st.checkbox("Family History")
    treatment = st.selectbox("Treatment", ["Yes", "No"])
    work_interfere = st.selectbox("Work Interfere", ["Sometimes", "Never", "Often"])
    no_employees = st.selectbox("No. of Employees", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.checkbox("Remote Work")
    tech_company = st.checkbox("Tech Company")
    benefits = st.selectbox("Benefits", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Care Options", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Wellness Program", ["Yes", "No", "Don't know"])
    seek_help = st.selectbox("Seek Help", ["Yes", "No", "Don't know"])
    anonymity = st.selectbox("Anonymity", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Leave", ["Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
    mental_health_consequence = st.selectbox("Mental Health Consequence", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Physical Health Consequence", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Coworkers", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Supervisor", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Mental Health Interview", ["Yes", "No", "Maybe"])
    phys_health_interview = st.selectbox("Physical Health Interview", ["Yes", "No", "Maybe"])
    mental_vs_physical = st.selectbox("Mental vs Physical", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Obs Consequence", ["Yes", "No"])

    # Convert Timestamp to string for consistent DataFrame creation
    Timestamp = datetime.strftime(Timestamp, "%Y-%m-%d %H:%M:%S")

    # Create a new data point
    new_data = pd.DataFrame({
        "Timestamp": [Timestamp],
        "Age": [Age],
        "Gender": [Gender],
        "Country": [Country],
        "state": [state],
        "self_employed": [self_employed],
        "family_history": [family_history],
        "treatment": [treatment],
        "work_interfere": [work_interfere],
        "no_employees": [no_employees],
        "remote_work": [remote_work],
        "tech_company": [tech_company],
        "benefits": [benefits],
        "care_options": [care_options],
        "wellness_program": [wellness_program],
        "seek_help": [seek_help],
        "anonymity": [anonymity],
        "leave": [leave],
        "mental_health_consequence": [mental_health_consequence],
        "phys_health_consequence": [phys_health_consequence],
        "coworkers": [coworkers],
        "supervisor": [supervisor],
        "mental_health_interview": [mental_health_interview],
        "phys_health_interview": [phys_health_interview],
        "mental_vs_physical": [mental_vs_physical],
        "obs_consequence": [obs_consequence]
    })

    # Preprocess the new data
    new_data_transformed = preprocessor.transform(new_data.drop(columns=['treatment'], axis=1))

    # Make a prediction
    prediction = model.predict(new_data_transformed)[0]

    if st.button('Predict'):
        if prediction == 1:
            result = 'Yes'
        else:
            result = 'No'
        st.success(f'The prediction for treatment is: {result}')

if __name__ == '__main__':
    run()
