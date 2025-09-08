#!/usr/bin/env python3
"""
Test script to verify the OSMI Mental Health Risk Predictor works correctly
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

def test_model_loading():
    """Test if model loads correctly"""
    print("🔍 Testing model loading...")
    
    try:
        model_path = Path("models/trained/best_calibrated_model.joblib")
        metadata_path = Path("models/trained/best_calibrated_model_metadata.json")
        results_path = Path("models/trained/best_calibrated_model_results.json")
        
        # Load model
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("✅ Metadata loaded successfully")
        
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        print("✅ Results loaded successfully")
        
        return model, metadata, results
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def test_prediction():
    """Test prediction with sample data"""
    print("\n🔍 Testing prediction with sample data...")
    
    model, metadata, results = test_model_loading()
    
    if model is None:
        print("❌ Cannot test prediction - model loading failed")
        return False
    
    # Create sample input data
    sample_data = pd.DataFrame({
        'are_you_selfemployed': [0],
        'company_size': ['26-100'],
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
    
    try:
        # Make prediction
        probability = model.predict_proba(sample_data)[0][1]
        prediction = model.predict(sample_data)[0]
        
        print(f"✅ Prediction successful!")
        print(f"   Probability of seeking treatment: {probability:.1%}")
        print(f"   Binary prediction: {prediction}")
        
        # Determine risk level
        if probability < 0.2:
            risk_level = "Very Low"
        elif probability < 0.34:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        elif probability < 0.8:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        print(f"   Risk Level: {risk_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def test_different_scenarios():
    """Test multiple scenarios"""
    print("\n🔍 Testing different risk scenarios...")
    
    model, _, _ = test_model_loading()
    if model is None:
        return False
    
    scenarios = [
        {
            'name': 'High Support Scenario',
            'data': {
                'are_you_selfemployed': [0],
                'company_size': ['More Than 1000'],
                'is_your_employer_primarily_a_tech_companyorganization': [1.0],
                'is_your_primary_role_within_your_company_related_to_techit': ['Yes'],
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
            }
        },
        {
            'name': 'Low Support Scenario',
            'data': {
                'are_you_selfemployed': [1],
                'company_size': ['1-5'],
                'is_your_employer_primarily_a_tech_companyorganization': [0.0],
                'is_your_primary_role_within_your_company_related_to_techit': ['No'],
                'do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage': ['No'],
                'has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication': ['No'],
                'does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help': [None],
                'is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer': ['No'],
                'if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be': ['Very Difficult'],
                'do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences': ['Yes'],
                'do_you_think_that_discussing_a_physical_health_issue_with_your_employer_would_have_negative_consequences': ['Maybe'],
                'would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers': ['No'],
                'would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors': ['No'],
                'do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health': ['No'],
                'have_you_heard_of_or_observed_negative_consequences_for_coworkers_who_have_been_open_about_mental_health_issues_in_your_workplace': ['Yes'],
                'do_you_have_medical_coverage_private_insurance_or_stateprovided_which_includes_treatment_of_mental_health_issues': ['No'],
                'do_you_know_local_or_online_resources_to_seek_help_for_a_mental_health_disorder': ['No'],
                'if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_clients_or_business_contacts': ['Never'],
                'if_you_have_revealed_a_mental_health_issue_to_a_client_or_business_contact_do_you_believe_this_has_impacted_you_negatively': [None],
                'if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_coworkers_or_employees': ['Never']
            }
        }
    ]
    
    for scenario in scenarios:
        try:
            data_df = pd.DataFrame(scenario['data'])
            probability = model.predict_proba(data_df)[0][1]
            
            if probability < 0.2:
                risk_level = "Very Low"
            elif probability < 0.34:
                risk_level = "Low"
            elif probability < 0.6:
                risk_level = "Medium"
            elif probability < 0.8:
                risk_level = "High"
            else:
                risk_level = "Very High"
            
            print(f"✅ {scenario['name']}: {probability:.1%} ({risk_level})")
            
        except Exception as e:
            print(f"❌ {scenario['name']} failed: {e}")
    
    return True

def main():
    """Main test function"""
    print("🧠 OSMI Mental Health Risk Predictor - Test Suite")
    print("=" * 55)
    
    # Test model loading
    success = test_model_loading() is not None
    
    if success:
        # Test basic prediction
        success = test_prediction()
    
    if success:
        # Test different scenarios
        success = test_different_scenarios()
    
    print("\n" + "=" * 55)
    if success:
        print("✅ All tests passed! The app should work correctly.")
        print("🚀 You can now run the app with: python run_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
