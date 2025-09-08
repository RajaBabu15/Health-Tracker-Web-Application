#!/usr/bin/env python3
"""
Simple test script with specific cases for testing the Streamlit app
with real data where the model predicts high likelihood of treatment seeking.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_app_with_real_cases():
    """Test the app with real high-risk cases from the dataset"""
    print("🧠 Mental Health Risk Predictor - App Test Cases")
    print("=" * 55)
    
    # Load model
    print("🔍 Loading model...")
    try:
        model_path = Path("models/trained/best_calibrated_model.joblib")
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load dataset
    print("🔍 Loading dataset...")
    try:
        data_path = Path("data/processed/osmi_processed.csv")
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded: {df.shape[0]} rows")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Find some high-risk cases
    print("🔍 Finding high-risk cases...")
    df_clean = df.dropna(subset=['treatment_sought'])
    X = df_clean.drop(['treatment_sought'], axis=1)
    y = df_clean['treatment_sought']
    
    # Get predictions
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    
    # Find high-risk cases
    high_risk_mask = probabilities > 0.7
    high_risk_indices = np.where(high_risk_mask)[0]
    
    print(f"📊 Found {len(high_risk_indices)} high-risk cases (>70% probability)")
    
    if len(high_risk_indices) == 0:
        print("❌ No high-risk cases found")
        return
    
    # Test cases for the Streamlit app
    print("\n🚀 STREAMLIT APP TEST CASES")
    print("=" * 55)
    print("Copy these values into the Streamlit app to test predictions:\n")
    
    # Test different types of cases
    test_cases = [
        ("High Risk - Treatment Seeker", 0),
        ("High Risk - Non-Treatment", 1), 
        ("Very High Risk", 2)
    ]
    
    for case_name, idx in test_cases:
        if idx >= len(high_risk_indices):
            continue
            
        case_idx = high_risk_indices[idx]
        case_data = df_clean.iloc[case_idx]
        prob = probabilities[case_idx]
        actual = case_data['treatment_sought']
        
        # Get risk level
        if prob < 0.2:
            risk_level = "🟢 Very Low"
        elif prob < 0.34:
            risk_level = "🟡 Low"
        elif prob < 0.6:
            risk_level = "🟠 Medium"
        elif prob < 0.8:
            risk_level = "🔴 High"
        else:
            risk_level = "⚫ Very High"
        
        print(f"**{case_name}**")
        print(f"Expected: {prob:.1%} probability ({risk_level})")
        print(f"Reality: {'Actually sought treatment' if actual == 1.0 else 'Actually did NOT seek treatment'}\n")
        
        print("📋 Enter these values in the app:")
        
        # Map to app format
        values = {
            "Are you self-employed?": "Yes" if case_data.get('are_you_selfemployed', 0) == 1 else "No",
            "Company size": case_data.get('company_size', 'Unknown'),
            "Is employer primarily a tech company?": "Yes" if case_data.get('is_your_employer_primarily_a_tech_companyorganization') == 1.0 else "No" if case_data.get('is_your_employer_primarily_a_tech_companyorganization') == 0.0 else "Not applicable",
            "Know mental health care options?": case_data.get('do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage', 'Unknown'),
            "Employer discussed mental health?": case_data.get('has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication', 'Unknown'),
            "Anonymity protected?": case_data.get('is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer', 'Unknown'),
            "Mental health leave difficulty": case_data.get('if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be', 'Unknown'),
            "Mental health consequences?": case_data.get('do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences', 'Unknown'),
            "Physical health consequences?": case_data.get('do_you_think_that_discussing_a_physical_health_issue_with_your_employer_would_have_negative_consequences', 'Unknown'),
            "Comfortable with coworkers?": case_data.get('would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers', 'Unknown'),
            "Comfortable with supervisors?": case_data.get('would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors', 'Unknown')
        }
        
        for question, answer in values.items():
            if pd.notna(answer) and str(answer) != 'nan' and answer != 'Unknown':
                print(f"  • {question}: {answer}")
        
        print("\n" + "-" * 50 + "\n")
    
    # Summary
    print("💡 HOW TO TEST:")
    print("1. Run the Streamlit app: python run_app.py")
    print("2. Enter the values from each test case above")
    print("3. Click 'Predict Treatment Likelihood'")
    print("4. Compare the app's prediction with the expected values")
    print("5. The predictions should match within a few percentage points")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("• Small differences (<5%) are normal due to rounding")
    print("• The app uses the same model, so predictions should be very close")
    print("• Pay attention to cases where someone didn't actually seek treatment")
    print("• This helps understand the model's limitations")

def quick_model_test():
    """Quick test to verify model predictions"""
    print("\n🧪 QUICK MODEL VERIFICATION")
    print("=" * 35)
    
    try:
        model = joblib.load("models/trained/best_calibrated_model.joblib")
        
        # Create a simple test case
        test_data = pd.DataFrame({
            'are_you_selfemployed': [0],
            'company_size': ['More Than 1000'],
            'is_your_employer_primarily_a_tech_companyorganization': [1.0],
            'is_your_primary_role_within_your_company_related_to_techit': [None],
            'do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage': ['Yes'],
            'has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication': ['Yes'],
            'does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help': [None],
            'is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer': ['Yes'],
            'if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be': ['Very Easy'],
            'do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences': ['No'],
            'do_you_think_that_discussing_a_physical_health_issue_with_your_employer_would_have_negative_consequences': ['No'],
            'would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers': ['Yes'],
            'would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors': ['Yes'],
            'do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health': ['Yes'],
            'have_you_heard_of_or_observed_negative_consequences_for_coworkers_who_have_been_open_about_mental_health_issues_in_your_workplace': ['No'],
            'do_you_have_medical_coverage_private_insurance_or_stateprovided_which_includes_treatment_of_mental_health_issues': ['Yes'],
            'do_you_know_local_or_online_resources_to_seek_help_for_a_mental_health_disorder': ['Yes'],
            'if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_clients_or_business_contacts': [None],
            'if_you_have_revealed_a_mental_health_issue_to_a_client_or_business_contact_do_you_believe_this_has_impacted_you_negatively': [None],
            'if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_coworkers_or_employees': [None]
        })
        
        prob = model.predict_proba(test_data)[0][1]
        pred = model.predict(test_data)[0]
        
        print(f"✅ Model test successful!")
        print(f"   Sample prediction: {prob:.1%} probability")
        print(f"   Classification: {'Would seek treatment' if pred == 1 else 'Would not seek treatment'}")
        
        if prob > 0.5:
            print(f"   Risk Level: 🔴 High Risk")
        else:
            print(f"   Risk Level: 🟡 Low Risk")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")

if __name__ == "__main__":
    test_app_with_real_cases()
    quick_model_test()
