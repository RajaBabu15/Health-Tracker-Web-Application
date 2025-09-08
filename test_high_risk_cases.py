#!/usr/bin/env python3
"""
Test high-risk cases and validate the Streamlit app functionality
with real data values that the model predicts as likely to seek treatment.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

def load_model_and_data():
    """Load the trained model and original dataset"""
    print("🔍 Loading model and dataset...")
    
    try:
        model_path = Path("models/trained/best_calibrated_model.joblib")
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        
        data_path = Path("data/processed/osmi_processed.csv")
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return model, df
        
    except Exception as e:
        print(f"❌ Error loading model/data: {e}")
        return None, None

def find_high_risk_cases(model, df):
    """Find cases where the model predicts high likelihood of treatment seeking"""
    print("\n🔍 Finding high-risk predictions...")
    
    # Remove rows with missing target values
    df_clean = df.dropna(subset=['treatment_sought'])
    
    # Prepare input data (remove target column)
    X = df_clean.drop(['treatment_sought'], axis=1)
    y_actual = df_clean['treatment_sought']
    
    # Get predictions for all cases
    try:
        probabilities = model.predict_proba(X)[:, 1]  # Probability of seeking treatment
        predictions = model.predict(X)
        
        # Create results dataframe
        results_df = df_clean.copy()
        results_df['predicted_probability'] = probabilities
        results_df['predicted_class'] = predictions
        
        # Find high-risk cases (probability > 70%)
        high_risk_cases = results_df[results_df['predicted_probability'] > 0.7]
        
        print(f"📊 Found {len(high_risk_cases)} high-risk cases (>70% probability)")
        
        return high_risk_cases
        
    except Exception as e:
        print(f"❌ Error making batch predictions: {e}")
        return None

def analyze_high_risk_cases(high_risk_cases):
    """Analyze the characteristics of high-risk cases"""
    print("\n📊 ANALYZING HIGH-RISK CASES")
    print("=" * 50)
    
    if high_risk_cases is None or len(high_risk_cases) == 0:
        print("❌ No high-risk cases to analyze")
        return
    
    # Basic statistics
    print(f"Total high-risk cases: {len(high_risk_cases)}")
    print(f"Average predicted probability: {high_risk_cases['predicted_probability'].mean():.1%}")
    
    # Accuracy on high-risk cases
    correct_high_risk = (high_risk_cases['predicted_class'] == high_risk_cases['treatment_sought']).sum()
    accuracy_high_risk = correct_high_risk / len(high_risk_cases)
    print(f"Accuracy on high-risk cases: {accuracy_high_risk:.1%} ({correct_high_risk}/{len(high_risk_cases)})")
    
    # Split by actual treatment status
    actually_sought = high_risk_cases[high_risk_cases['treatment_sought'] == 1.0]
    actually_didnt_seek = high_risk_cases[high_risk_cases['treatment_sought'] == 0.0]
    
    print(f"\nBreakdown:")
    print(f"  Actually sought treatment: {len(actually_sought)} cases")
    print(f"  Actually didn't seek treatment: {len(actually_didnt_seek)} cases")
    
    if len(actually_didnt_seek) > 0:
        print(f"\n⚠️ FALSE POSITIVES: {len(actually_didnt_seek)} cases predicted high-risk but didn't seek treatment")
        print("This suggests the model may be over-predicting in some cases")
    
    # Show some key patterns
    print("\n🔍 Common characteristics in high-risk cases:")
    
    # Analyze key features
    key_features = [
        'are_you_selfemployed',
        'company_size',
        'is_your_employer_primarily_a_tech_companyorganization',
        'do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage',
        'has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication'
    ]
    
    for feature in key_features:
        if feature in high_risk_cases.columns:
            value_counts = high_risk_cases[feature].value_counts()
            if len(value_counts) > 0:
                top_value = value_counts.index[0]
                top_count = value_counts.iloc[0]
                percentage = (top_count / len(high_risk_cases)) * 100
                print(f"  {feature}: {top_value} ({percentage:.1f}%)")

def test_specific_high_risk_cases(high_risk_cases):
    """Test specific high-risk cases with detailed analysis"""
    print("\n🧪 TESTING SPECIFIC HIGH-RISK CASES")
    print("=" * 50)
    
    if high_risk_cases is None or len(high_risk_cases) == 0:
        print("❌ No cases to test")
        return []
    
    # Select interesting cases
    test_cases = []
    
    # Case 1: Highest probability case
    highest_prob_case = high_risk_cases.loc[high_risk_cases['predicted_probability'].idxmax()]
    test_cases.append(("Highest Probability", highest_prob_case))
    
    # Case 2: High probability but didn't actually seek treatment (false positive)
    false_positives = high_risk_cases[high_risk_cases['treatment_sought'] == 0.0]
    if len(false_positives) > 0:
        fp_case = false_positives.iloc[0]
        test_cases.append(("False Positive", fp_case))
    
    # Case 3: High probability and did seek treatment (true positive)
    true_positives = high_risk_cases[high_risk_cases['treatment_sought'] == 1.0]
    if len(true_positives) > 0:
        tp_case = true_positives.iloc[0]
        test_cases.append(("True Positive", tp_case))
    
    app_test_data = []
    
    for case_name, case_data in test_cases:
        print(f"\n🧪 {case_name}")
        print("-" * 30)
        
        # Basic info
        actual = "Yes" if case_data['treatment_sought'] == 1.0 else "No"
        predicted_prob = case_data['predicted_probability']
        predicted_class = "Yes" if case_data['predicted_class'] == 1.0 else "No"
        
        print(f"Actual treatment sought: {actual}")
        print(f"Predicted probability: {predicted_prob:.1%}")
        print(f"Predicted class: {predicted_class}")
        
        # Risk level
        if predicted_prob < 0.2:
            risk_level = "🟢 Very Low"
        elif predicted_prob < 0.34:
            risk_level = "🟡 Low"
        elif predicted_prob < 0.6:
            risk_level = "🟠 Medium"
        elif predicted_prob < 0.8:
            risk_level = "🔴 High"
        else:
            risk_level = "⚫ Very High"
        
        print(f"Risk level: {risk_level}")
        
        # Show app-compatible format
        print("\n📱 Streamlit App Format:")
        app_data = {
            'Self-employed': "Yes" if case_data['are_you_selfemployed'] == 1 else "No",
            'Company size': case_data['company_size'],
            'Tech company': "Yes" if case_data['is_your_employer_primarily_a_tech_companyorganization'] == 1.0 else "No" if case_data['is_your_employer_primarily_a_tech_companyorganization'] == 0.0 else "Not applicable",
            'Know mental health options': case_data['do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage'],
            'Employer discussed MH': case_data['has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication'],
            'Anonymity protected': case_data['is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer'],
            'Leave difficulty': case_data['if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be'],
            'MH consequences': case_data['do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences'],
            'Physical consequences': case_data['do_you_think_that_discussing_a_physical_health_issue_with_your_employer_would_have_negative_consequences'],
            'Coworker comfort': case_data['would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers'],
            'Supervisor comfort': case_data['would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors']
        }
        
        for key, value in app_data.items():
            if pd.notna(value):
                print(f"  {key}: {value}")
        
        app_test_data.append({
            'name': case_name,
            'data': app_data,
            'expected_prob': predicted_prob,
            'expected_risk': risk_level,
            'actual_treatment': actual
        })

def generate_app_test_instructions(app_test_data):
    """Generate instructions for testing the Streamlit app"""
    print("\n🚀 STREAMLIT APP TEST INSTRUCTIONS")
    print("=" * 50)
    
    print("Copy and paste these values into the Streamlit app to verify predictions:\n")
    
    for i, test_case in enumerate(app_test_data, 1):
        print(f"**Test Case {i}: {test_case['name']}**")
        print(f"Expected: {test_case['expected_prob']:.1%} probability, {test_case['expected_risk']}")
        print(f"Reality: Actually {'sought' if test_case['actual_treatment'] == 'Yes' else 'did not seek'} treatment\n")
        
        print("Input values:")
        for key, value in test_case['data'].items():
            if pd.notna(value) and value != "Not specified":
                print(f"  {key}: {value}")
        
        print("\n" + "-" * 40 + "\n")

def main():
    """Main test function"""
    print("🧠 OSMI Mental Health Risk Predictor - High Risk Cases Analysis")
    print("=" * 65)
    
    # Load model and data
    model, df = load_model_and_data()
    
    if model is None or df is None:
        print("❌ Cannot proceed without model and data")
        return False
    
    # Find high-risk cases
    high_risk_cases = find_high_risk_cases(model, df)
    
    # Analyze high-risk cases
    analyze_high_risk_cases(high_risk_cases)
    
    # Test specific cases
    app_test_data = test_specific_high_risk_cases(high_risk_cases)
    
    # Generate app test instructions
    if app_test_data:
        generate_app_test_instructions(app_test_data)
    else:
        print("\n❌ No test data generated for app testing")
    
    print("\n" + "=" * 65)
    print("🏁 Analysis completed!")
    print("\n💡 Key Findings:")
    print("• High-risk cases show specific patterns that the model learned")
    print("• False positives reveal model limitations")
    print("• Use the test cases above to validate the Streamlit app")
    print("• Compare app predictions with expected values")
    
    return True

if __name__ == "__main__":
    main()
