"""
Configuration settings for OSMI Mental Health Risk Prediction System
"""
import os
from pathlib import Path
from typing import Dict, Any, List

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model paths
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
MODEL_ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# Plot directories
EDA_PLOTS_DIR = PLOTS_DIR / "eda"
MODEL_PLOTS_DIR = PLOTS_DIR / "model_evaluation"
FEATURE_PLOTS_DIR = PLOTS_DIR / "feature_importance"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR, 
                  RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                  TRAINED_MODELS_DIR, MODEL_ARTIFACTS_DIR,
                  EDA_PLOTS_DIR, MODEL_PLOTS_DIR, FEATURE_PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    "raw_csv_file": "osmi_raw.csv",
    "raw_json_file": "osmi_raw.json",
    "processed_file": "osmi_processed.csv",
    "features_file": "feature_engineered.csv",
    "train_file": "train_data.csv",
    "test_file": "test_data.csv",
    "validation_file": "validation_data.csv"
}

# Model configuration
MODEL_CONFIG = {
    "target_column": "treatment_sought",
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "early_stopping_rounds": 100,
    "n_jobs": -1
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "age_bins": [18, 25, 30, 35, 40, 50, 100],
    "age_labels": ["18-24", "25-29", "30-34", "35-39", "40-49", "50+"],
    "categorical_encoding": "target",  # target, onehot, label
    "handle_missing": "impute",  # impute, drop, flag
    "rare_category_threshold": 0.05,
    "feature_selection_k": 20
}

# LightGBM default parameters
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42
}

# XGBoost default parameters
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "random_state": 42
}

# Optuna hyperparameter optimization
OPTUNA_CONFIG = {
    "n_trials": 100,
    "timeout": 3600,  # 1 hour
    "direction": "maximize",  # maximize ROC AUC
    "pruner": "MedianPruner",
    "sampler": "TPESampler"
}

# Risk scoring configuration
RISK_CONFIG = {
    "low_threshold": 0.3,
    "medium_threshold": 0.7,
    "high_threshold": 0.9,
    "risk_levels": ["Low", "Medium", "High", "Critical"],
    "calibration_method": "isotonic"  # platt, isotonic
}

# Visualization configuration
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "whitegrid",
    "color_palette": "husl",
    "font_scale": 1.2
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
    "max_request_size": 1024 * 1024,  # 1MB
    "timeout": 30
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "OSMI Mental Health Risk Predictor",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "osmi_system.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# Feature importance thresholds
SHAP_CONFIG = {
    "max_display": 20,
    "sample_size": 1000,  # for SHAP calculation efficiency
    "interaction_analysis": True,
    "waterfall_max_features": 10
}

# Model evaluation metrics
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall", 
    "f1",
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score_loss"
]

# Treatment recommendation mapping
RECOMMENDATION_CONFIG = {
    "low_risk": [
        "Continue current wellness practices",
        "Consider mindfulness or meditation apps",
        "Maintain regular exercise routine",
        "Connect with supportive friends/family"
    ],
    "medium_risk": [
        "Schedule check-in with mental health professional",
        "Explore employee assistance programs (EAP)",
        "Practice stress management techniques",
        "Consider therapy or counseling services",
        "Review work-life balance strategies"
    ],
    "high_risk": [
        "Seek immediate mental health support",
        "Contact employee mental health hotline",
        "Schedule appointment with therapist/psychiatrist",
        "Inform trusted supervisor about need for support",
        "Consider taking mental health leave if needed"
    ],
    "critical_risk": [
        "Contact mental health crisis line immediately",
        "Reach out to emergency mental health services",
        "Inform emergency contact about situation",
        "Do not delay seeking professional help",
        "Consider immediate medical attention if needed"
    ]
}

# Feature descriptions for interpretability
FEATURE_DESCRIPTIONS = {
    "age": "Age of respondent in years",
    "gender": "Gender identity of respondent",
    "family_history": "Family history of mental illness",
    "treatment": "Currently receiving mental health treatment",
    "work_interfere": "Does mental health interfere with work",
    "no_employees": "Company size by number of employees",
    "remote_work": "Works remotely",
    "tech_company": "Works at a tech company",
    "benefits": "Employer provides mental health benefits",
    "care_options": "Aware of mental health care options",
    "wellness_program": "Employer has wellness program",
    "seek_help": "Resources available to seek help",
    "anonymity": "Anonymity protected when seeking help",
    "leave": "Ease of taking mental health leave",
    "mental_health_consequence": "Negative consequences for mental health issues",
    "phys_health_consequence": "Negative consequences for physical health issues",
    "coworkers": "Comfortable discussing mental health with coworkers",
    "supervisor": "Comfortable discussing mental health with supervisor"
}

# Export key configurations for easy import
__all__ = [
    'BASE_DIR', 'DATA_DIR', 'MODELS_DIR', 'PLOTS_DIR',
    'DATA_CONFIG', 'MODEL_CONFIG', 'FEATURE_CONFIG',
    'LIGHTGBM_PARAMS', 'XGBOOST_PARAMS', 'OPTUNA_CONFIG',
    'RISK_CONFIG', 'VIZ_CONFIG', 'API_CONFIG', 'STREAMLIT_CONFIG',
    'SHAP_CONFIG', 'EVALUATION_METRICS', 'RECOMMENDATION_CONFIG',
    'FEATURE_DESCRIPTIONS'
]
