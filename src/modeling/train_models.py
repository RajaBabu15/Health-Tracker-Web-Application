"""
Production-ready OSMI Mental Health Risk Prediction Training Pipeline

Modular architecture with single-purpose functions for:
- Data loading and cleaning
- Feature engineering and preprocessing  
- Model training and hyperparameter optimization
- Model evaluation and calibration
- Artifact persistence and metadata tracking
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import logging

# Add project root to sys.path for module discovery
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import lightgbm as lgb
import xgboost as xgb
import optuna
import shap

from config.config import (
    PROCESSED_DATA_DIR, TRAINED_MODELS_DIR, MODEL_PLOTS_DIR, FEATURE_PLOTS_DIR,
    DATA_CONFIG, MODEL_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
LEAKY_FEATURE = 'does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help'
MISSING_THRESHOLD = 0.8
OPTUNA_TRIALS = 100
OPTUNA_TIMEOUT = 1800


# Data loading and cleaning functions

def load_raw_data() -> pd.DataFrame:
    """Load processed dataset from disk."""
    data_path = PROCESSED_DATA_DIR / DATA_CONFIG["processed_file"]
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove target missings, leaky features, and high-missing columns."""
    logger.info("Starting data cleaning")
    
    # Remove missing targets
    initial_rows = len(df)
    df_clean = df.dropna(subset=['treatment_sought'])
    logger.info(f"Removed {initial_rows - len(df_clean)} rows with missing targets")
    
    # Remove known leaky feature
    if LEAKY_FEATURE in df_clean.columns:
        df_clean = df_clean.drop(columns=[LEAKY_FEATURE])
        logger.info(f"Removed leaky feature: {LEAKY_FEATURE}")
    
    # Remove high-missing features
    high_missing_cols = _identify_high_missing_features(df_clean)
    if high_missing_cols:
        df_clean = df_clean.drop(columns=high_missing_cols)
        logger.info(f"Removed {len(high_missing_cols)} high-missing features")
    
    logger.info(f"Final cleaned shape: {df_clean.shape}")
    return df_clean


def _identify_high_missing_features(df: pd.DataFrame) -> List[str]:
    """Identify features with missing rate above threshold."""
    high_missing = []
    for col in df.columns:
        if col == 'treatment_sought':
            continue
        missing_rate = df[col].isnull().sum() / len(df)
        if missing_rate > MISSING_THRESHOLD:
            high_missing.append(col)
    return high_missing
    

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/validation/test splits."""
    logger.info("Splitting data")
    
    X = df.drop(columns=['treatment_sought'])
    y = df['treatment_sought'].astype(int)
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.12, random_state=RANDOM_STATE, stratify=y
    )
    
    # Second split: train vs validation  
    val_ratio = 165 / len(X_temp)  # Target ~165 validation samples
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    # Combine features and targets
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    logger.info(f"Splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def create_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    logger.info("Creating preprocessor")
    
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Features - Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}")
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor
    
# Model training functions

def train_baseline_logistic(X_train: pd.DataFrame, y_train: pd.Series, 
                           X_val: pd.DataFrame, y_val: pd.Series,
                           preprocessor: ColumnTransformer) -> Dict[str, Any]:
    """Train baseline logistic regression with preprocessing pipeline."""
    logger.info("Training baseline logistic regression")
    
    try:
        # Suppress numerical warnings for logistic regression
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.linear_model')
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    class_weight='balanced',
                    solver='liblinear',  # More stable than lbfgs for this data
                    max_iter=2000,       # Increased iterations
                    C=0.1,               # Regularization to prevent overflow
                    random_state=RANDOM_STATE
                ))
            ])
            
            pipeline.fit(X_train, y_train)
            
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_pred = pipeline.predict(X_val)
        
        results = {
            'model': pipeline,
            'val_roc_auc': roc_auc_score(y_val, val_proba),
            'val_pr_auc': average_precision_score(y_val, val_proba),
            'val_brier': brier_score_loss(y_val, val_proba),
            'val_f1': f1_score(y_val, val_pred)
        }
        
        logger.info(f"Baseline ROC AUC: {results['val_roc_auc']:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Baseline training failed: {e}")
        raise
    

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series,
                  preprocessor: ColumnTransformer) -> Dict[str, Any]:
    """Train LightGBM with early stopping."""
    logger.info("Training LightGBM")
    
    # Suppress LightGBM feature name warnings
    import warnings
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    
    try:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            learning_rate=0.05,
            n_estimators=1000,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        
        model.fit(
            X_train_processed, y_train,
            eval_set=[(X_val_processed, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        val_proba = model.predict_proba(X_val_processed)[:, 1]
        val_pred = model.predict(X_val_processed)
        
        results = {
            'model': model,
            'preprocessor': preprocessor,
            'val_roc_auc': roc_auc_score(y_val, val_proba),
            'val_pr_auc': average_precision_score(y_val, val_proba),
            'val_brier': brier_score_loss(y_val, val_proba),
            'val_f1': f1_score(y_val, val_pred),
            'best_iteration': model.best_iteration_
        }
        
        logger.info(f"LightGBM ROC AUC: {results['val_roc_auc']:.4f}, Best iter: {results['best_iteration']}")
        return results
        
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        raise


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series,
                 preprocessor: ColumnTransformer) -> Dict[str, Any]:
    """Train XGBoost with class balancing."""
    logger.info("Training XGBoost")
    
    try:
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            learning_rate=0.05,
            n_estimators=1000,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            early_stopping_rounds=50,
            eval_metric='logloss'
        )
        
        model.fit(
            X_train_processed, y_train,
            eval_set=[(X_val_processed, y_val)],
            verbose=False
        )
        
        val_proba = model.predict_proba(X_val_processed)[:, 1]
        val_pred = model.predict(X_val_processed)
        
        results = {
            'model': model,
            'preprocessor': preprocessor,
            'val_roc_auc': roc_auc_score(y_val, val_proba),
            'val_pr_auc': average_precision_score(y_val, val_proba),
            'val_brier': brier_score_loss(y_val, val_proba),
            'val_f1': f1_score(y_val, val_pred),
            'best_iteration': model.best_iteration
        }
        
        logger.info(f"XGBoost ROC AUC: {results['val_roc_auc']:.4f}, Best iter: {results['best_iteration']}")
        return results
        
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        raise


# Model evaluation functions

def evaluate_model(model: Any, preprocessor: ColumnTransformer, 
                  X_test: pd.DataFrame, y_test: pd.Series,
                  model_name: str, is_pipeline: bool = False) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    logger.info(f"Evaluating {model_name}")
    
    try:
        if is_pipeline:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:
            X_test_processed = preprocessor.transform(X_test)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            y_pred = model.predict(X_test_processed)
        
        # Core metrics
        results = {
            'model_name': model_name,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
        
        # Precision@k (top 10% alerts)
        k = max(1, int(0.1 * len(y_test)))
        top_k_indices = np.argsort(y_pred_proba)[-k:]
        precision_at_k = y_test.iloc[top_k_indices].mean()
        results['precision_at_10pct'] = precision_at_k
        
        logger.info(f"{model_name} - ROC AUC: {results['roc_auc']:.4f}, Precision@10%: {precision_at_k:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed for {model_name}: {e}")
        raise


def calibrate_model(model: Any, preprocessor: ColumnTransformer,
                   X_val: pd.DataFrame, y_val: pd.Series) -> Any:
    """Calibrate model probabilities using isotonic regression."""
    logger.info("Calibrating model")
    
    try:
        X_val_processed = preprocessor.transform(X_val)
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(X_val_processed, y_val)
        return calibrated
    except Exception as e:
        logger.error(f"Model calibration failed: {e}")
        raise


def find_optimal_threshold(model: Any, preprocessor: ColumnTransformer,
                          X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Find threshold that maximizes F1 score."""
    try:
        X_val_processed = preprocessor.transform(X_val)
        y_proba = model.predict_proba(X_val_processed)[:, 1]
        
        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1_scores.append(f1_score(y_val, y_pred))
        
        best_threshold = thresholds[np.argmax(f1_scores)]
        logger.info(f"Optimal threshold: {best_threshold:.3f}")
        return best_threshold
        
    except Exception as e:
        logger.error(f"Threshold optimization failed: {e}")
        raise


# Artifact persistence functions

def save_model_artifacts(model: Any, preprocessor: ColumnTransformer, 
                        model_name: str, results: Dict[str, Any],
                        metadata: Dict[str, Any]) -> None:
    """Save model, preprocessor, results, and metadata."""
    logger.info(f"Saving artifacts for {model_name}")
    
    try:
        # Ensure directories exist
        TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model and preprocessor as pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        model_path = TRAINED_MODELS_DIR / f'{model_name}.joblib'
        joblib.dump(pipeline, model_path, compress=3)
        
        # Save results
        results_path = TRAINED_MODELS_DIR / f'{model_name}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metadata
        metadata_path = TRAINED_MODELS_DIR / f'{model_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Validation checks
        roc_auc = results['roc_auc']
        brier_score = results['brier_score']
        
        if roc_auc >= 0.75:
            logger.info(f"ROC AUC ({roc_auc:.4f}) meets production target")
        elif roc_auc >= 0.65:
            logger.warning(f"ROC AUC ({roc_auc:.4f}) meets minimum threshold")
        else:
            logger.error(f"ROC AUC ({roc_auc:.4f}) below minimum threshold")
        
        if brier_score <= 0.18:
            logger.info(f"Brier score ({brier_score:.4f}) excellent calibration")
        elif brier_score <= 0.25:
            logger.info(f"Brier score ({brier_score:.4f}) good calibration")
        else:
            logger.warning(f"Brier score ({brier_score:.4f}) needs improvement")
        
        logger.info(f"Artifacts saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to save artifacts: {e}")
        raise


def create_training_metadata(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create metadata dictionary for training run."""
    return {
        'timestamp': datetime.now().isoformat(),
        'random_state': RANDOM_STATE,
        'optuna_trials': OPTUNA_TRIALS,
        'optuna_timeout': OPTUNA_TIMEOUT,
        'missing_threshold': MISSING_THRESHOLD,
        'leaky_feature_removed': LEAKY_FEATURE,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'parameters': params or {}
    }


def optimize_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series,
                             preprocessor: ColumnTransformer) -> Dict[str, Any]:
    """Optimize LightGBM hyperparameters using Optuna."""
    logger.info("Starting hyperparameter optimization")
    
    # Suppress warnings during hyperparameter optimization
    import warnings
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    warnings.filterwarnings('ignore', message='.*Optuna.*')
    
    X_train_processed = preprocessor.transform(X_train)
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': RANDOM_STATE,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train_processed, y_train, cv=cv, scoring='roc_auc')
        return scores.mean()
    
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
        
        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'boosting_type': 'gbdt', 
            'random_state': RANDOM_STATE,
            'verbose': -1
        })
        
        optimized_model = lgb.LGBMClassifier(**best_params)
        optimized_model.fit(X_train_processed, y_train)
        
        logger.info(f"Best CV ROC AUC: {study.best_value:.4f}")
        
        return {
            'model': optimized_model,
            'preprocessor': preprocessor,
            'best_params': best_params,
            'cv_roc_auc': study.best_value
        }
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        raise

def run() -> Dict[str, Any]:
    """Run the complete training pipeline with modular components."""
    # Suppress general warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    
    try:
        # Load and clean data
        df = load_raw_data()
        df_clean = clean_data(df)

        # Split data
        train_df, val_df, test_df = split_data(df_clean)

        # Separate features and labels
        X_train, y_train = train_df.drop(columns=['treatment_sought']), train_df['treatment_sought'].astype(int)
        X_val, y_val = val_df.drop(columns=['treatment_sought']), val_df['treatment_sought'].astype(int)
        X_test, y_test = test_df.drop(columns=['treatment_sought']), test_df['treatment_sought'].astype(int)

        # Preprocessor
        preprocessor = create_preprocessor(X_train)

        # Train models
        baseline = train_baseline_logistic(X_train, y_train, X_val, y_val, preprocessor)
        lgbm = train_lightgbm(X_train, y_train, X_val, y_val, preprocessor)
        xgbm = train_xgboost(X_train, y_train, X_val, y_val, preprocessor)

        # Choose best of LGBM vs XGB
        best = lgbm if lgbm['val_roc_auc'] >= xgbm['val_roc_auc'] else xgbm
        best_model = best['model']

        # Hyperparameter optimization (LightGBM only)
        optimized = optimize_hyperparameters(X_train, y_train, preprocessor)
        if optimized['cv_roc_auc'] >= best['val_roc_auc']:
            best_model = optimized['model']
            best_name = 'optimized_lightgbm'
            # Add optimized model to results for evaluation
            optimized_val_proba = best_model.predict_proba(preprocessor.transform(X_val))[:, 1]
            optimized_val_pred = best_model.predict(preprocessor.transform(X_val))
            optimized_results = {
                'model': best_model,
                'preprocessor': preprocessor,
                'val_roc_auc': roc_auc_score(y_val, optimized_val_proba),
                'val_pr_auc': average_precision_score(y_val, optimized_val_proba),
                'val_brier': brier_score_loss(y_val, optimized_val_proba),
                'val_f1': f1_score(y_val, optimized_val_pred)
            }
        else:
            best_name = 'lightgbm' if best is lgbm else 'xgboost'
            optimized_results = None

        # Calibrate and threshold
        calibrated = calibrate_model(best_model, preprocessor, X_val, y_val)
        threshold = find_optimal_threshold(calibrated, preprocessor, X_val, y_val)

        # Evaluate all models (baseline is a pipeline)
        results_all: Dict[str, Dict[str, Any]] = {}
        results_all['baseline_logistic'] = evaluate_model(baseline['model'], preprocessor, X_test, y_test, 'baseline_logistic', is_pipeline=True)
        results_all['lightgbm'] = evaluate_model(lgbm['model'], preprocessor, X_test, y_test, 'lightgbm')
        results_all['xgboost'] = evaluate_model(xgbm['model'], preprocessor, X_test, y_test, 'xgboost')
        
        # Add optimized model if it was selected
        if optimized_results is not None:
            results_all['optimized_lightgbm'] = evaluate_model(optimized_results['model'], preprocessor, X_test, y_test, 'optimized_lightgbm')
            
        results_all['calibrated_best'] = evaluate_model(calibrated, preprocessor, X_test, y_test, 'calibrated_best')

        # Metadata
        metadata = create_training_metadata()

        # Save artifacts
        save_model_artifacts(calibrated, preprocessor, 'best_calibrated_model', results_all['calibrated_best'], metadata)
        save_model_artifacts(best_model, preprocessor, best_name, results_all[best_name], metadata)

        # Return concise summary for CI logs
        summary = {
            'best_model_name': best_name,
            'optimal_threshold': float(threshold),
            'metrics': results_all['calibrated_best'],
            'dataset_info': {
                'train_size': len(X_train), 'val_size': len(X_val), 'test_size': len(X_test)
            }
        }
        logger.info("Training pipeline completed successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    try:
        out = run()
        logger.info("=== TRAINING SUMMARY ===")
        logger.info(json.dumps(out, indent=2))
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import sys
        sys.exit(1)
