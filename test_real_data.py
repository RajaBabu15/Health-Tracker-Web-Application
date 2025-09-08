#!/usr/bin/env python3
"""
Test the OSMI Mental Health Risk Predictor using real data from the original dataset.
This script tests with actual cases where people did or didn't seek treatment.
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
        # Load model
        model_path = Path("models/trained/best_calibrated_model.joblib")
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        
        # Load original dataset
        data_path = Path("data/processed/osmi_processed.csv")
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return model, df
        
    except Exception as e:
        print(f"❌ Error loading model/data: {e}")
        return None, None

def test_real_cases(model, df):
    """Test model predictions on real cases from the dataset"""
    print("\n🔍 Testing with real cases from the dataset...")
    
    # Remove rows with missing target values
    df_clean = df.dropna(subset=['treatment_sought'])
    
    # Get cases where people DID seek treatment (treatment_sought = 1)
    treatment_cases = df_clean[df_clean['treatment_sought'] == 1.0]
    
    # Get cases where people did NOT seek treatment (treatment_sought = 0)
    no_treatment_cases = df_clean[df_clean['treatment_sought'] == 0.0]
    
    print(f"📊 Found {len(treatment_cases)} cases who sought treatment")
    print(f"📊 Found {len(no_treatment_cases)} cases who didn't seek treatment")
    
    # Test 2 cases from each group
    test_cases = []
    
    # Add 2 treatment-seeking cases
    if len(treatment_cases) >= 2:
        test_cases.extend([
            ("Treatment Case 1", treatment_cases.iloc[0]),
            ("Treatment Case 2", treatment_cases.iloc[10])  # Different case
        ])
    
    # Add 2 non-treatment cases
    if len(no_treatment_cases) >= 2:
        test_cases.extend([
            ("No Treatment Case 1", no_treatment_cases.iloc[0]),
            ("No Treatment Case 2", no_treatment_cases.iloc[10])  # Different case
        ])
    
    results = []
    
    for case_name, case_data in test_cases:
        print(f"\n🧪 Testing: {case_name}")
        print("=" * 50)
        
        # Prepare input data (remove target column)
        input_data = case_data.drop(['treatment_sought']).to_frame().T
        actual_treatment = case_data['treatment_sought']
        
        try:
            # Make prediction
            probability = model.predict_proba(input_data)[0][1]
            predicted_class = model.predict(input_data)[0]
            
            # Determine risk level
            if probability < 0.2:
                risk_level = "Very Low"
                risk_emoji = "🟢"
            elif probability < 0.34:
                risk_level = "Low"
                risk_emoji = "🟡"
            elif probability < 0.6:
                risk_level = "Medium" 
                risk_emoji = "🟠"
            elif probability < 0.8:
                risk_level = "High"
                risk_emoji = "🔴"
            else:
                risk_level = "Very High"
                risk_emoji = "⚫"
            
            # Check if prediction matches reality
            correct = (predicted_class == actual_treatment)
            match_emoji = "✅" if correct else "❌"
            
            print(f"{match_emoji} Actual Treatment Sought: {'Yes' if actual_treatment == 1 else 'No'}")
            print(f"   Predicted Probability: {probability:.1%}")
            print(f"   Predicted Class: {'Yes' if predicted_class == 1 else 'No'}")
            print(f"   Risk Level: {risk_emoji} {risk_level}")
            print(f"   Prediction Correct: {'Yes' if correct else 'No'}")
            
            # Show some key features that influenced the prediction
            print("   Key Features:")
            feature_names = input_data.columns
            for i, feature in enumerate(feature_names[:5]):  # Show first 5 features
                value = input_data.iloc[0, i]
                if pd.notna(value):
                    print(f"     • {feature}: {value}")
            
            results.append({
                'case': case_name,
                'actual': actual_treatment,
                'predicted_prob': probability,
                'predicted_class': predicted_class,
                'risk_level': risk_level,
                'correct': correct
            })
            
        except Exception as e:
            print(f"❌ Prediction failed for {case_name}: {e}")
            continue
    
    return results

def analyze_results(results):
    """Analyze the test results"""
    print("\n📊 RESULTS SUMMARY")
    print("=" * 50)
    
    if not results:
        print("❌ No successful predictions to analyze")
        return
    
    # Overall accuracy
    correct_predictions = sum(1 for r in results if r['correct'])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions
    
    print(f"Overall Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    # Breakdown by actual treatment status
    treatment_cases = [r for r in results if r['actual'] == 1.0]
    no_treatment_cases = [r for r in results if r['actual'] == 0.0]
    
    if treatment_cases:
        treatment_correct = sum(1 for r in treatment_cases if r['correct'])
        print(f"Treatment Cases: {treatment_correct}/{len(treatment_cases)} correct")
        avg_prob_treatment = np.mean([r['predicted_prob'] for r in treatment_cases])
        print(f"  Average predicted probability: {avg_prob_treatment:.1%}")
    
    if no_treatment_cases:
        no_treatment_correct = sum(1 for r in no_treatment_cases if r['correct'])
        print(f"No Treatment Cases: {no_treatment_correct}/{len(no_treatment_cases)} correct")
        avg_prob_no_treatment = np.mean([r['predicted_prob'] for r in no_treatment_cases])
        print(f"  Average predicted probability: {avg_prob_no_treatment:.1%}")

def test_edge_cases(model, df):
    """Test some interesting edge cases"""
    print("\n🔍 Testing Edge Cases...")
    print("=" * 50)
    
    df_clean = df.dropna(subset=['treatment_sought'])
    
    # Find cases with high support but didn't seek treatment
    high_support_no_treatment = df_clean[
        (df_clean['treatment_sought'] == 0.0) &
        (df_clean['do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage'] == 'Yes') &
        (df_clean['has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication'] == 'Yes')
    ]
    
    # Find cases with low support but did seek treatment
    low_support_treatment = df_clean[
        (df_clean['treatment_sought'] == 1.0) &
        (df_clean['do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage'] == 'No') &
        (df_clean['do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences'] == 'Yes')
    ]
    
    edge_cases = []
    
    if len(high_support_no_treatment) > 0:
        edge_cases.append(("High Support, No Treatment", high_support_no_treatment.iloc[0]))
    
    if len(low_support_treatment) > 0:
        edge_cases.append(("Low Support, But Sought Treatment", low_support_treatment.iloc[0]))
    
    for case_name, case_data in edge_cases:
        print(f"\n🧪 Edge Case: {case_name}")
        print("-" * 30)
        
        input_data = case_data.drop(['treatment_sought']).to_frame().T
        actual_treatment = case_data['treatment_sought']
        
        try:
            probability = model.predict_proba(input_data)[0][1]
            predicted_class = model.predict(input_data)[0]
            
            print(f"Actual: {'Sought Treatment' if actual_treatment == 1 else 'No Treatment'}")
            print(f"Predicted Probability: {probability:.1%}")
            print(f"Model says: {'Would seek treatment' if predicted_class == 1 else 'Would not seek treatment'}")
            
            # This shows how well the model handles counter-intuitive cases
            if case_name == "High Support, No Treatment" and probability > 0.5:
                print("🤔 Model thinks high support = treatment seeking (may miss individual factors)")
            elif case_name == "Low Support, But Sought Treatment" and probability < 0.5:
                print("🤔 Model thinks low support = no treatment (may miss personal motivation)")
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")

def main():
    """Main test function"""
    print("🧠 OSMI Mental Health Risk Predictor - Real Data Test")
    print("=" * 60)
    
    # Load model and data
    model, df = load_model_and_data()
    
    if model is None or df is None:
        print("❌ Cannot proceed without model and data")
        return False
    
    # Test real cases
    results = test_real_cases(model, df)
    
    # Analyze results
    analyze_results(results)
    
    # Test edge cases
    test_edge_cases(model, df)
    
    print("\n" + "=" * 60)
    print("🏁 Testing completed!")
    print("\n💡 Key Insights:")
    print("• Check how well the model identifies people who actually sought treatment")
    print("• Look for patterns in cases where the model was wrong")
    print("• Edge cases show model limitations and biases")
    
    return True

if __name__ == "__main__":
    main()
