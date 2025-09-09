"""
🧠 OSMI Mental Health Risk Predictor - Streamlit App

A user-friendly web application to predict likelihood of seeking mental health treatment
based on workplace and personal factors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global UI styling (professional, minimal)
CUSTOM_CSS = """
<style>
/***** Typography *****/
html, body, [class^=block-container] { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji", sans-serif; }
:root { --brand-primary: #1f497d; --brand-accent: #2e6cc5; --muted: #6b7280; --border: #e5e7eb; --success: #198754; --warning: #d97706; --danger: #b91c1c; }

/***** Headings and spacing *****/
h1, h2, h3 { color: var(--brand-primary); letter-spacing: 0.1px; }
h1 { font-weight: 700; }
h2, h3 { font-weight: 600; }

/***** Cards and sections *****/
.section-card { border: 1px solid var(--border); border-radius: 10px; padding: 16px 16px 8px; margin: 8px 0 16px; background: #ffffff; }
.small-muted { color: var(--muted); font-size: 12px; }

/***** Badges *****/
.badge { display: inline-block; padding: 2px 6px; border-radius: 999px; font-size: 11px; font-weight: 600; margin-left: 6px; vertical-align: middle; border: 1px solid transparent; }
.tier-critical { color: #7c2d12; background: #fff7ed; border-color: #fdba74; }
.tier-high { color: #1e4620; background: #ecfdf5; border-color: #86efac; }
.tier-medium { color: #1e3a8a; background: #eff6ff; border-color: #93c5fd; }

/***** Labels *****/
label, .stSelectbox label { font-weight: 500; color: #111827; }

/***** Sidebar headings *****/
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #111827; }

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Feature importance rankings based on our comprehensive analysis
FEATURE_IMPORTANCE_RANKING = {
    'does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help': {
        'rank': 1, 'score': 40, 'tier': 'critical', 'color': '#FF4B4B', 'badge': '🏆 #1 MOST IMPORTANT'
    },
    'has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication': {
        'rank': 2, 'score': 26, 'tier': 'critical', 'color': '#FF6B35', 'badge': '🥈 #2 CRITICAL'
    },
    'is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer': {
        'rank': 3, 'score': 23, 'tier': 'critical', 'color': '#F7931E', 'badge': '🥉 #3 CRITICAL'
    },
    'company_size': {
        'rank': 4, 'score': 21, 'tier': 'critical', 'color': '#FFB800', 'badge': '⭐ #4 CRITICAL'
    },
    'do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health': {
        'rank': 5, 'score': 19, 'tier': 'high', 'color': '#37B24D', 'badge': '📈 #5 HIGH IMPACT'
    },
    'would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors': {
        'rank': 6, 'score': 14, 'tier': 'high', 'color': '#51CF66', 'badge': '📊 #6 HIGH IMPACT'
    },
    'do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage': {
        'rank': 7, 'score': 12, 'tier': 'high', 'color': '#69DB7C', 'badge': '📋 #7 HIGH IMPACT'
    },
    'if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be': {
        'rank': 8, 'score': 9, 'tier': 'medium', 'color': '#74C0FC', 'badge': '📌 #8 MEDIUM'
    },
    'would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers': {
        'rank': 9, 'score': 9, 'tier': 'medium', 'color': '#91A7FF', 'badge': '📌 #9 MEDIUM'
    },
    'are_you_selfemployed': {
        'rank': 10, 'score': 9, 'tier': 'medium', 'color': '#B197FC', 'badge': '📌 #10 MEDIUM'
    }
}

# Toggle to show/hide feature-importance annotations; default off
if 'show_importance' not in st.session_state:
    st.session_state.show_importance = False


def get_feature_importance_info(feature_key):
    """Get importance information for a feature"""
    return FEATURE_IMPORTANCE_RANKING.get(feature_key, {'rank': None, 'score': 0, 'tier': 'low', 'color': '#868E96', 'badge': ''})

def importance_enabled():
    return bool(st.session_state.get('show_importance', False))


def create_importance_badge(feature_key):
    """Create an importance badge for a feature"""
    if not importance_enabled():
        return ""
    info = get_feature_importance_info(feature_key)
    if info['rank'] and info['rank'] <= 10:
        return f" (#{info['rank']})"
    return ""

def render_question_with_badge(label_text, feature_key):
    """Render question label with optional importance badge"""
    if not importance_enabled():
        return label_text
    info = get_feature_importance_info(feature_key)
    if info['rank'] and info['rank'] <= 10:
        tier_class = f"tier-{info['tier']}"
        badge_html = f"<span class='badge {tier_class}'>#{info['rank']}</span>"
        st.markdown(f"**{label_text}** {badge_html}", unsafe_allow_html=True)
        return "_"  # Hidden label for selectbox
    return label_text


def get_importance_help_text(feature_key, base_help):
    """Get enhanced help text with importance information"""
    info = get_feature_importance_info(feature_key)
    if info['rank'] and info['rank'] <= 10:
        importance_text = f"\n\n🎯 FEATURE IMPORTANCE: #{info['rank']} (Score: {info['score']})\n"
        if info['tier'] == 'critical':
            importance_text += "🔥 CRITICAL FACTOR - This has major impact on prediction!"
        elif info['tier'] == 'high':
            importance_text += "📈 HIGH IMPACT - This strongly influences the outcome!"
        else:
            importance_text += "📊 MODERATE IMPACT - This influences the prediction."
        return base_help + importance_text
    return base_help

# Load model and metadata
@st.cache_resource
def load_model():
    """Load the trained model and metadata"""
    try:
        model_path = Path("models/trained/best_calibrated_model.joblib")
        metadata_path = Path("models/trained/best_calibrated_model_metadata.json")
        results_path = Path("models/trained/best_calibrated_model_results.json")
        
        model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        with open(results_path, 'r') as f:
            results = json.load(f)
            
        return model, metadata, results
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Helper functions
def get_risk_level(probability, threshold=0.34):
    """Convert probability to risk level"""
    if probability < 0.2:
        return "Very Low", "#4CAF50", "🟢"
    elif probability < threshold:
        return "Low", "#8BC34A", "🟡"  
    elif probability < 0.6:
        return "Medium", "#FF9800", "🟠"
    elif probability < 0.8:
        return "High", "#FF5722", "🔴"
    else:
        return "Very High", "#D32F2F", "⚫"

def get_recommendations(risk_level):
    """Get recommendations based on risk level"""
    recommendations = {
        "Very Low": [
            "🌟 Continue your current wellness practices",
            "🧘 Consider mindfulness or meditation apps", 
            "🏃 Maintain regular exercise routine",
            "👥 Connect with supportive friends/family"
        ],
        "Low": [
            "🌟 Continue your current wellness practices",
            "🧘 Consider mindfulness or meditation apps",
            "📚 Explore mental health resources available to you",
            "👥 Build supportive relationships at work"
        ],
        "Medium": [
            "👨‍⚕️ Consider scheduling check-in with mental health professional",
            "🏢 Explore employee assistance programs (EAP)",
            "🧘 Practice stress management techniques",
            "⚖️ Review work-life balance strategies"
        ],
        "High": [
            "🚨 Consider seeking mental health support soon",
            "📞 Contact employee mental health resources",
            "👨‍⚕️ Schedule appointment with therapist/counselor",
            "👔 Consider discussing support needs with supervisor"
        ],
        "Very High": [
            "🆘 Seek mental health support as soon as possible",
            "📞 Contact mental health crisis line if needed",
            "👨‍⚕️ Schedule urgent appointment with mental health professional",
            "👥 Inform trusted contacts about your situation"
        ]
    }
    return recommendations.get(risk_level, recommendations["Medium"])

def create_probability_gauge(probability, risk_level, color):
    """Create a gauge chart for probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Treatment Seeking Likelihood", 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'size': 30}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 20], 'color': '#E8F5E8'},
                {'range': [20, 34], 'color': '#FFF3E0'},
                {'range': [34, 60], 'color': '#FFF0E6'},
                {'range': [60, 80], 'color': '#FFEBEE'},
                {'range': [80, 100], 'color': '#FFCDD2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
    return fig

# Main app
def main():
    model, metadata, results = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if the model files exist.")
        return
    
    # Header
    st.title("Mental Health Treatment Predictor")
    st.markdown("""
    Predict the likelihood of seeking mental health treatment based on workplace and personal factors.
    
    This tool uses machine learning to assess treatment-seeking likelihood based on responses from the 
    Open Sourcing Mental Illness (OSMI) survey data.
    """)
    
    # Sidebar with model info
    with st.sidebar:
        st.header("Model Information")
        if results:
            st.metric("Accuracy", f"{results['accuracy']:.1%}")
            st.metric("ROC AUC", f"{results['roc_auc']:.3f}")
            st.metric("F1 Score", f"{results['f1']:.3f}")
        
        st.markdown("---")
        st.markdown("**Details**")
        st.write(f"Training Date: {metadata['timestamp'][:10] if metadata else 'Unknown'}")
        st.write("Model: Calibrated LightGBM")
        st.write("Dataset: OSMI Mental Health Survey")
        
        # UI preferences
        st.markdown("---")
        st.subheader("Display Options")
        st.session_state.show_importance = st.checkbox("Show feature-importance annotations", value=False, help="Toggle minimal badges next to questions to indicate their relative importance.")
        
        if importance_enabled():
            st.markdown("**Feature Importance Legend**")
            st.markdown("- Badge #1 – #3: Critical impact")
            st.markdown("- Badge #4 – #7: High impact")
            st.markdown("- Badge #8 – #10: Medium impact")
            st.caption("Based on comprehensive statistical analysis")
        
        # Risk level legend
        st.markdown("---")
        st.markdown("**Risk Levels**")
        st.write("Very Low (0–20%)")
        st.write("Low (20–34%)")
        st.write("Medium (34–60%)")
        st.write("High (60–80%)")
        st.write("Very High (80–100%)")
        
        # Test cases section
        st.markdown("---")
        st.markdown("**Sample Cases**")
        st.caption("Test with real, validated cases from the dataset.")
        
        # Buttons for people who actually sought treatment
        st.markdown("**Sought Treatment**")
        
        # Define test cases from our real data testing
        positive_cases = {
            "Case A": {
                "are_you_selfemployed": 0,
                "company_size": "6-25",
                "is_your_employer_primarily_a_tech_companyorganization": 1.0,
                "is_your_primary_role_within_your_company_related_to_techit": None,
                "do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage": "Yes",
                "has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication": "Yes",
                "does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help": "Yes",
                "is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer": "Yes",
                "if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be": "Somewhat Easy",
                "do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences": "No",
                "do_you_think_that_discussing_a_physical_health_issue_with_your_employer_would_have_negative_consequences": "No",
                "would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers": "Maybe",
                "would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors": "Yes",
                "do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health": None,
                "have_you_heard_of_or_observed_negative_consequences_for_coworkers_who_have_been_open_about_mental_health_issues_in_your_workplace": None,
                "do_you_have_medical_coverage_private_insurance_or_stateprovided_which_includes_treatment_of_mental_health_issues": None,
                "do_you_know_local_or_online_resources_to_seek_help_for_a_mental_health_disorder": None,
                "if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_clients_or_business_contacts": None,
                "if_you_have_revealed_a_mental_health_issue_to_a_client_or_business_contact_do_you_believe_this_has_impacted_you_negatively": None,
                "if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_coworkers_or_employees": None,
                "description": "Tech worker with good support - actually sought treatment"
            },
            "Case B": {
                "are_you_selfemployed": 0,
                "company_size": "100-500",
                "is_your_employer_primarily_a_tech_companyorganization": 0.0,
                "is_your_primary_role_within_your_company_related_to_techit": None,
                "do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage": "Yes",
                "has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication": "Yes",
                "does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help": "Yes",
                "is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer": "Yes",
                "if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be": "Somewhat Easy",
                "do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences": "No",
                "do_you_think_that_discussing_a_physical_health_issue_with_your_employer_would_have_negative_consequences": "No",
                "would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers": "No",
                "would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors": "Maybe",
                "do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health": None,
                "have_you_heard_of_or_observed_negative_consequences_for_coworkers_who_have_been_open_about_mental_health_issues_in_your_workplace": None,
                "do_you_have_medical_coverage_private_insurance_or_stateprovided_which_includes_treatment_of_mental_health_issues": None,
                "do_you_know_local_or_online_resources_to_seek_help_for_a_mental_health_disorder": None,
                "if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_clients_or_business_contacts": None,
                "if_you_have_revealed_a_mental_health_issue_to_a_client_or_business_contact_do_you_believe_this_has_impacted_you_negatively": None,
                "if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_coworkers_or_employees": None,
                "description": "Non-tech worker with excellent support - actually sought treatment"
            }
        }
        
        # Create buttons for positive cases
        col_red1, col_red2 = st.columns(2)
        with col_red1:
            if st.button("Case A", help=positive_cases["Case A"]['description'], key="test_case_A", use_container_width=True):
                for key, value in positive_cases["Case A"].items():
                    if key != 'description':
                        st.session_state[f"test_{key}"] = value
                st.rerun()
        
        with col_red2:
            if st.button("Case B", help=positive_cases["Case B"]['description'], key="test_case_B", use_container_width=True):
                for key, value in positive_cases["Case B"].items():
                    if key != 'description':
                        st.session_state[f"test_{key}"] = value
                st.rerun()
        
        # Buttons for people who did NOT seek treatment
        st.markdown("**Did Not Seek Treatment**")
        
        negative_cases = {
            "Case C": {
                "are_you_selfemployed": 0,
                "company_size": "26-100",
                "is_your_employer_primarily_a_tech_companyorganization": 1.0,
                "is_your_primary_role_within_your_company_related_to_techit": None,
                "do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage": None,
                "has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication": "No",
                "does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help": "No",
                "is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer": "Don't know",
                "if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be": "Very Difficult",
                "do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences": "No",
                "do_you_think_that_discussing_a_physical_health_issue_with_your_employer_would_have_negative_consequences": "No",
                "would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers": "Maybe",
                "would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors": "Maybe",
                "do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health": "Don't know",
                "have_you_heard_of_or_observed_negative_consequences_for_coworkers_who_have_been_open_about_mental_health_issues_in_your_workplace": "No",
                "do_you_have_medical_coverage_private_insurance_or_stateprovided_which_includes_treatment_of_mental_health_issues": None,
                "do_you_know_local_or_online_resources_to_seek_help_for_a_mental_health_disorder": None,
                "if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_clients_or_business_contacts": None,
                "if_you_have_revealed_a_mental_health_issue_to_a_client_or_business_contact_do_you_believe_this_has_impacted_you_negatively": None,
                "if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_coworkers_or_employees": None,
                "description": "Tech company with limited support - actually did NOT seek treatment"
            },
            "Case D": {
                "are_you_selfemployed": 0,
                "company_size": "26-100",
                "is_your_employer_primarily_a_tech_companyorganization": 1.0,
                "is_your_primary_role_within_your_company_related_to_techit": None,
                "do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage": "Yes",
                "has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication": "No",
                "does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help": "No",
                "is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer": "Don't know",
                "if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be": "Neither Easy Nor Difficult",
                "do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences": "No",
                "do_you_think_that_discussing_a_physical_health_issue_with_your_employer_would_have_negative_consequences": "No",
                "would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers": "Maybe",
                "would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors": "Maybe",
                "do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health": "Don't know",
                "have_you_heard_of_or_observed_negative_consequences_for_coworkers_who_have_been_open_about_mental_health_issues_in_your_workplace": "No",
                "do_you_have_medical_coverage_private_insurance_or_stateprovided_which_includes_treatment_of_mental_health_issues": None,
                "do_you_know_local_or_online_resources_to_seek_help_for_a_mental_health_disorder": None,
                "if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_clients_or_business_contacts": None,
                "if_you_have_revealed_a_mental_health_issue_to_a_client_or_business_contact_do_you_believe_this_has_impacted_you_negatively": None,
                "if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_coworkers_or_employees": None,
                "description": "Tech company with mixed support - actually did NOT seek treatment"
            }
        }
        
        # Create buttons for negative cases
        col_green1, col_green2 = st.columns(2)
        with col_green1:
            if st.button("Case C", help=negative_cases["Case C"]['description'], key="test_case_C", use_container_width=True):
                for key, value in negative_cases["Case C"].items():
                    if key != 'description':
                        st.session_state[f"test_{key}"] = value
                st.rerun()
        
        with col_green2:
            if st.button("Case D", help=negative_cases["Case D"]['description'], key="test_case_D", use_container_width=True):
                for key, value in negative_cases["Case D"].items():
                    if key != 'description':
                        st.session_state[f"test_{key}"] = value
                st.rerun()
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Please answer the following questions")
        
        # Feature-importance explanation (concise)
        if importance_enabled():
            st.caption("Feature-importance badges show which questions drive predictions the most.")
        
        # Employment Information
        st.subheader("Employment Information")
        
        col1a, col1b = st.columns(2)
        with col1a:
            # Get default values from session state if available
            default_self_employed = st.session_state.get("test_are_you_selfemployed", 0)
            default_company_size = st.session_state.get("test_company_size", "1-5")
            
            self_employed = st.selectbox(
                render_question_with_badge("Are you self-employed?", 'are_you_selfemployed'),
                options=[0, 1],
                index=[0, 1].index(default_self_employed),
                format_func=lambda x: "No" if x == 0 else "Yes",
                help=get_importance_help_text('are_you_selfemployed', "Select whether you are self-employed or work for a company")
            )
            
            company_size = st.selectbox(
                render_question_with_badge("Company size (number of employees)", 'company_size'),
                options=["1-5", "6-25", "26-100", "100-500", "500-1000", "More Than 1000"],
                index=["1-5", "6-25", "26-100", "100-500", "500-1000", "More Than 1000"].index(default_company_size) if default_company_size in ["1-5", "6-25", "26-100", "100-500", "500-1000", "More Than 1000"] else 0,
                help=get_importance_help_text('company_size', "Select your company's size by number of employees. Larger companies typically have better mental health support!")
            )
            
        with col1b:
            # Get default values from session state if available
            default_tech_company = st.session_state.get("test_is_your_employer_primarily_a_tech_companyorganization", 1.0)
            default_tech_role = st.session_state.get("test_is_your_primary_role_within_your_company_related_to_techit", None)
            
            tech_company = st.selectbox(
                "Is your employer primarily a tech company?",
                options=[1.0, 0.0, np.nan],
                index=[1.0, 0.0, np.nan].index(default_tech_company) if not pd.isna(default_tech_company) else 0,
                format_func=lambda x: "Yes" if x == 1.0 else ("No" if x == 0.0 else "Not applicable"),
                help="Select if your employer is primarily a technology company"
            )
            
            tech_role = st.selectbox(
                "Is your primary role related to tech/IT?",
                options=[None, "Yes", "No"],
                index=[None, "Yes", "No"].index(default_tech_role) if default_tech_role in [None, "Yes", "No"] else 0,
                format_func=lambda x: "Not specified" if x is None else x,
                help="Select if your primary role is technology or IT related"
            )
        
        # Mental Health Coverage & Support
        st.subheader("Mental Health Coverage & Support")
        
        col2a, col2b = st.columns(2)
        with col2a:
            # Get default values from session state if available
            default_know_options = st.session_state.get("test_do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage", "No")
            default_employer_discussion = st.session_state.get("test_has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication", "No")
            default_employer_resources = st.session_state.get("test_does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help", "No")
            default_anonymity = st.session_state.get("test_is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer", "Don't know")
            
            know_options = st.selectbox(
                render_question_with_badge("Do you know mental health care options under your coverage?", 'do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage'),
                options=["No", "Maybe", "Yes"],
                index=["No", "Maybe", "Yes"].index(default_know_options) if default_know_options in ["No", "Maybe", "Yes"] else 0,
                help=get_importance_help_text('do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage', "Do you know what mental health care options are available under your employer coverage?")
            )
            
            employer_discussion = st.selectbox(
                render_question_with_badge("Has employer formally discussed mental health?", 'has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication'),
                options=["No", "Yes"],
                index=["No", "Yes"].index(default_employer_discussion) if default_employer_discussion in ["No", "Yes"] else 0,
                help=get_importance_help_text('has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication', "Has your employer ever formally discussed mental health (e.g., wellness campaign)?")
            )
            
            employer_resources = st.selectbox(
                render_question_with_badge("Does employer offer mental health learning resources?", 'does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help'),
                options=["No", "Yes"],
                index=["No", "Yes"].index(default_employer_resources) if default_employer_resources in ["No", "Yes"] else 0,
                help=get_importance_help_text('does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help', "Does your employer offer resources to learn more about mental health? This is the MOST IMPORTANT predictor!")
            )
            
            anonymity = st.selectbox(
                render_question_with_badge("Is anonymity protected when seeking mental health treatment?", 'is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer'),
                options=["Don't know", "Yes", "No"],
                index=["Don't know", "Yes", "No"].index(default_anonymity) if default_anonymity in ["Don't know", "Yes", "No"] else 0,
                help=get_importance_help_text('is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer', "Is your anonymity protected if you use employer mental health resources?")
            )
            
        with col2b:
            # Get default values from session state if available
            default_leave_ease = st.session_state.get("test_if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be", "Very Easy")
            default_negative_consequences = st.session_state.get("test_do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences", "No")
            default_physical_consequences = st.session_state.get("test_do_you_think_that_discussing_a_physical_health_issue_with_your_employer_would_have_negative_consequences", "No")
            
            leave_ease = st.selectbox(
                "How easy would requesting mental health leave be?",
                options=["Very Easy", "Somewhat Easy", "Neither Easy Nor Difficult", 
                        "Somewhat Difficult", "Very Difficult"],
                index=["Very Easy", "Somewhat Easy", "Neither Easy Nor Difficult", "Somewhat Difficult", "Very Difficult"].index(default_leave_ease) if default_leave_ease in ["Very Easy", "Somewhat Easy", "Neither Easy Nor Difficult", "Somewhat Difficult", "Very Difficult"] else 0,
                help="If you needed mental health leave, how easy would requesting it be?"
            )
            
            negative_consequences = st.selectbox(
                "Would discussing mental health have negative consequences?",
                options=["No", "Maybe", "Yes"],
                index=["No", "Maybe", "Yes"].index(default_negative_consequences) if default_negative_consequences in ["No", "Maybe", "Yes"] else 0,
                help="Do you think discussing mental health with employer would have negative consequences?"
            )
            
            physical_consequences = st.selectbox(
                "Would discussing physical health have negative consequences?",
                options=["No", "Maybe", "Yes"],
                index=["No", "Maybe", "Yes"].index(default_physical_consequences) if default_physical_consequences in ["No", "Maybe", "Yes"] else 0,
                help="Do you think discussing physical health issues would have negative consequences?"
            )
        
        # Workplace Comfort & Culture
        st.subheader("Workplace Comfort & Culture")
        
        col3a, col3b = st.columns(2)
        with col3a:
            # Get default values from session state if available
            default_coworker_comfort = st.session_state.get("test_would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers", "Maybe")
            default_supervisor_comfort = st.session_state.get("test_would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors", "Maybe")
            
            coworker_comfort = st.selectbox(
                render_question_with_badge("Comfortable discussing mental health with coworkers?", 'would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers'),
                options=["Maybe", "Yes", "No"],
                index=["Maybe", "Yes", "No"].index(default_coworker_comfort) if default_coworker_comfort in ["Maybe", "Yes", "No"] else 0,
                help=get_importance_help_text('would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers', "Would you feel comfortable discussing mental health with coworkers?")
            )
            
            supervisor_comfort = st.selectbox(
                render_question_with_badge("Comfortable discussing mental health with supervisors?", 'would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors'),
                options=["Maybe", "Yes", "No"],
                index=["Maybe", "Yes", "No"].index(default_supervisor_comfort) if default_supervisor_comfort in ["Maybe", "Yes", "No"] else 0,
                help=get_importance_help_text('would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors', "Would you feel comfortable discussing mental health with direct supervisors?")
            )
            
        with col3b:
            # Get default values from session state if available
            default_employer_seriousness = st.session_state.get("test_do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health", "Don't know")
            default_observed_consequences = st.session_state.get("test_have_you_heard_of_or_observed_negative_consequences_for_coworkers_who_have_been_open_about_mental_health_issues_in_your_workplace", "No")
            
            employer_seriousness = st.selectbox(
                render_question_with_badge("Does employer take mental health as seriously as physical?", 'do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health'),
                options=["Don't know", "Yes", "No"],
                index=["Don't know", "Yes", "No"].index(default_employer_seriousness) if default_employer_seriousness in ["Don't know", "Yes", "No"] else 0,
                help=get_importance_help_text('do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health', "Do you feel your employer takes mental health as seriously as physical health?")
            )
            
            observed_consequences = st.selectbox(
                "Observed negative consequences for coworkers?",
                options=["No", "Yes"],
                index=["No", "Yes"].index(default_observed_consequences) if default_observed_consequences in ["No", "Yes"] else 0,
                help="Have you observed negative consequences for coworkers open about mental health?"
            )
        
        # Additional Information
        st.subheader("Additional Information")
        
        col4a, col4b = st.columns(2)
        with col4a:
            # Get default values from session state if available
            default_medical_coverage = st.session_state.get("test_do_you_have_medical_coverage_private_insurance_or_stateprovided_which_includes_treatment_of_mental_health_issues", None)
            default_know_resources = st.session_state.get("test_do_you_know_local_or_online_resources_to_seek_help_for_a_mental_health_disorder", None)
            
            medical_coverage = st.selectbox(
                "Do you have medical coverage for mental health?",
                options=[None, "Yes", "No"],
                index=[None, "Yes", "No"].index(default_medical_coverage) if default_medical_coverage in [None, "Yes", "No"] else 0,
                format_func=lambda x: "Not specified" if x is None else x,
                help="Do you have medical coverage that includes mental health treatment?"
            )
            
            know_resources = st.selectbox(
                "Do you know local/online mental health resources?",
                options=[None, "Yes", "No"],
                index=[None, "Yes", "No"].index(default_know_resources) if default_know_resources in [None, "Yes", "No"] else 0,
                format_func=lambda x: "Not specified" if x is None else x,
                help="Do you know local or online resources for mental health help?"
            )
            
        with col4b:
            # Get default values from session state if available
            default_reveal_clients = st.session_state.get("test_if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_clients_or_business_contacts", None)
            default_client_impact = st.session_state.get("test_if_you_have_revealed_a_mental_health_issue_to_a_client_or_business_contact_do_you_believe_this_has_impacted_you_negatively", None)
            
            reveal_clients = st.selectbox(
                "Would you reveal mental health issues to clients?",
                options=[None, "Sometimes, If It Comes Up", "Never", "Yes"],
                index=[None, "Sometimes, If It Comes Up", "Never", "Yes"].index(default_reveal_clients) if default_reveal_clients in [None, "Sometimes, If It Comes Up", "Never", "Yes"] else 0,
                format_func=lambda x: "Not specified" if x is None else x,
                help="Would you reveal mental health diagnosis to clients/business contacts?"
            )
            
            client_impact = st.selectbox(
                "Has revealing to clients impacted you negatively?",
                options=[None, "Maybe", "No", "Yes"],
                index=[None, "Maybe", "No", "Yes"].index(default_client_impact) if default_client_impact in [None, "Maybe", "No", "Yes"] else 0,
                format_func=lambda x: "Not specified" if x is None else x,
                help="If you've revealed to clients, has it impacted you negatively?"
            )
        
        # Get default value from session state if available
        default_reveal_coworkers = st.session_state.get("test_if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_coworkers_or_employees", None)
        
        reveal_coworkers = st.selectbox(
            "Would you reveal mental health issues to coworkers?",
            options=[None, "Sometimes, If It Comes Up", "Never", "Yes"],
            index=[None, "Sometimes, If It Comes Up", "Never", "Yes"].index(default_reveal_coworkers) if default_reveal_coworkers in [None, "Sometimes, If It Comes Up", "Never", "Yes"] else 0,
            format_func=lambda x: "Not specified" if x is None else x,
            help="Would you reveal mental health diagnosis to coworkers/employees?"
        )
        
        # Add some spacing and buttons
        st.markdown("---")
        
        # Create two columns for buttons
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            # Clear form button
            if st.button("🗑️ Clear Form", help="Reset all fields to default values", use_container_width=True):
                # Clear all test session state variables
                keys_to_remove = [key for key in st.session_state.keys() if key.startswith('test_')]
                for key in keys_to_remove:
                    del st.session_state[key]
                st.rerun()
        
        with btn_col2:
            # Predict button
            predict_button = st.button("Predict Treatment Likelihood", type="primary", use_container_width=True)
        
        if predict_button:
            # Prepare input data to match EXACT training data format
            input_data = pd.DataFrame({
                'are_you_selfemployed': [int(self_employed)],
                'company_size': [str(company_size)],
                'is_your_employer_primarily_a_tech_companyorganization': [float(tech_company) if not pd.isna(tech_company) else np.nan],
                'is_your_primary_role_within_your_company_related_to_techit': [float(1.0) if tech_role == "Yes" else (float(0.0) if tech_role == "No" else np.nan)],
                'do_you_know_the_options_for_mental_health_care_available_under_your_employerprovided_coverage': [know_options],
                'has_your_employer_ever_formally_discussed_mental_health_for_example_as_part_of_a_wellness_campaign_or_other_official_communication': [employer_discussion],
                'does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help': [employer_resources],
                'is_your_anonymity_protected_if_you_choose_to_take_advantage_of_mental_health_or_substance_abuse_treatment_resources_provided_by_your_employer': [anonymity],
                'if_a_mental_health_issue_prompted_you_to_request_a_medical_leave_from_work_asking_for_that_leave_would_be': [leave_ease],
                'do_you_think_that_discussing_a_mental_health_disorder_with_your_employer_would_have_negative_consequences': [negative_consequences],
                'do_you_think_that_discussing_a_physical_health_issue_with_your_employer_would_have_negative_consequences': [physical_consequences],
                'would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers': [coworker_comfort],
                'would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors': [supervisor_comfort],
                'do_you_feel_that_your_employer_takes_mental_health_as_seriously_as_physical_health': [employer_seriousness],
                'have_you_heard_of_or_observed_negative_consequences_for_coworkers_who_have_been_open_about_mental_health_issues_in_your_workplace': [observed_consequences],
                'do_you_have_medical_coverage_private_insurance_or_stateprovided_which_includes_treatment_of_mental_health_issues': [np.nan if medical_coverage is None else (float(1.0) if medical_coverage == "Yes" else float(0.0))],
                'do_you_know_local_or_online_resources_to_seek_help_for_a_mental_health_disorder': [np.nan if know_resources is None else (float(1.0) if know_resources == "Yes" else float(0.0))],
                'if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_clients_or_business_contacts': [np.nan if reveal_clients is None else (float(1.0) if reveal_clients == "Yes" else (float(0.5) if reveal_clients == "Sometimes, If It Comes Up" else float(0.0)))],
                'if_you_have_revealed_a_mental_health_issue_to_a_client_or_business_contact_do_you_believe_this_has_impacted_you_negatively': [np.nan if client_impact is None else (float(1.0) if client_impact == "Yes" else (float(0.5) if client_impact == "Maybe" else float(0.0)))],
                'if_you_have_been_diagnosed_or_treated_for_a_mental_health_disorder_do_you_ever_reveal_this_to_coworkers_or_employees': [np.nan if reveal_coworkers is None else (float(1.0) if reveal_coworkers == "Yes" else (float(0.5) if reveal_coworkers == "Sometimes, If It Comes Up" else float(0.0)))]
            })
            
            try:
                # Make prediction with better error handling
                warnings.filterwarnings('ignore')
                prediction_proba = model.predict_proba(input_data)
                probability = prediction_proba[0][1]
                risk_level, color, emoji = get_risk_level(probability)
                
                # Store results in session state for display in right column
                st.session_state.probability = probability
                st.session_state.risk_level = risk_level
                st.session_state.color = color
                st.session_state.emoji = emoji
                st.session_state.show_results = True
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error(f"Debug info: Input data shape: {input_data.shape}")
                st.error(f"Debug info: Input data types: {input_data.dtypes.to_dict()}")
                st.session_state.show_results = False
    
    # Right column - Results
    with col2:
        if hasattr(st.session_state, 'show_results') and st.session_state.show_results:
            st.header("Results")
            
            # Gauge chart
            fig = create_probability_gauge(st.session_state.probability, st.session_state.risk_level, st.session_state.color)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk level display
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: {st.session_state.color}20; border-radius: 10px; margin: 10px 0;">
                <h2 style="color: {st.session_state.color}; margin: 0;">{st.session_state.emoji} {st.session_state.risk_level} Risk</h2>
                <p style="font-size: 18px; margin: 5px 0;">{st.session_state.probability:.1%} likelihood</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("Recommendations")
            recommendations = get_recommendations(st.session_state.risk_level)
            for rec in recommendations:
                st.write(f"• {rec}")
            
            # Important note
            st.markdown("---")
            st.warning("This prediction is based on statistical patterns and should not replace professional medical advice. If you're experiencing mental health concerns, please consult with a qualified mental health professional.")
            
        else:
            st.header("Prediction Results")
            st.info("Fill out the form and click 'Predict Treatment Likelihood' to see your results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 12px;">
        <p>OSMI Mental Health Risk Predictor</p>
        <p>Data source: <a href="https://osmihelp.org/" target="_blank">Open Sourcing Mental Illness (OSMI)</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
