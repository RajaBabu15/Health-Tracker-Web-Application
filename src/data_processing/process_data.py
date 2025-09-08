"""
Data Processing Module for OSMI Mental Health Risk Prediction

This module handles:
1. Data loading and initial cleaning
2. Feature engineering and selection
3. Target variable creation
4. Train/test/validation split
5. Data preprocessing for ML models
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_CONFIG, 
    MODEL_CONFIG, FEATURE_CONFIG
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSMIDataProcessor:
    """
    Comprehensive data processor for OSMI mental health survey data
    """
    
    def __init__(self):
        self.raw_csv_path = RAW_DATA_DIR / DATA_CONFIG["raw_csv_file"]
        self.raw_json_path = RAW_DATA_DIR / DATA_CONFIG["raw_json_file"]
        self.processed_path = PROCESSED_DATA_DIR / DATA_CONFIG["processed_file"]
        
        self.df = None
        self.feature_names = []
        self.target_column = MODEL_CONFIG["target_column"]
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV and JSON files"""
        logger.info("Loading raw data...")
        
        try:
            # Load CSV data
            df_csv = pd.read_csv(self.raw_csv_path, low_memory=False)
            logger.info(f"Loaded CSV data: {df_csv.shape}")
            
            # Load JSON data for additional context (optional)
            try:
                with open(self.raw_json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                logger.info("Loaded JSON metadata successfully")
            except Exception as e:
                logger.warning(f"Could not load JSON data: {e}")
                json_data = None
            
            self.df = df_csv
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and standardize the raw data"""
        logger.info("Cleaning data...")
        
        if self.df is None:
            self.load_data()
        
        # Make a copy to avoid modifying original
        df = self.df.copy()
        
        # Clean column names
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        # Handle missing values represented as strings
        df = df.replace(['NA', 'N/A', '', 'nan', 'NaN', 'null'], np.nan)
        
        # Clean specific columns
        df = self._clean_age_column(df)
        df = self._clean_gender_column(df)
        df = self._clean_categorical_columns(df)
        
        self.df = df
        logger.info(f"Cleaned data shape: {df.shape}")
        return df
    
    def create_target_variable(self) -> pd.DataFrame:
        """Create the target variable for treatment seeking behavior"""
        logger.info("Creating target variable...")
        
        if self.df is None:
            self.clean_data()
        
        df = self.df.copy()
        
        # Find treatment-related columns
        treatment_columns = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in 
                  ['treatment', 'sought', 'professional', 'help'])
        ]
        
        logger.info(f"Found treatment columns: {treatment_columns}")
        
        # Use the most direct treatment question
        primary_treatment_col = None
        for col in treatment_columns:
            if 'sought treatment' in col.lower() or 'mental health professional' in col.lower():
                primary_treatment_col = col
                break
        
        if primary_treatment_col is None and treatment_columns:
            primary_treatment_col = treatment_columns[0]
        
        if primary_treatment_col:
            # Create binary target
            df[self.target_column] = df[primary_treatment_col].apply(
                self._normalize_yes_no_response
            )
            logger.info(f"Created target variable from: {primary_treatment_col}")
            logger.info(f"Target distribution:\n{df[self.target_column].value_counts()}")
        else:
            logger.error("Could not find suitable treatment column")
            raise ValueError("No treatment column found for target creation")
        
        self.df = df
        return df
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features from raw survey responses"""
        logger.info("Engineering features...")
        
        if self.df is None or self.target_column not in self.df.columns:
            self.create_target_variable()
        
        df = self.df.copy()
        
        # Age-based features
        df = self._create_age_features(df)
        
        # Workplace features
        df = self._create_workplace_features(df)
        
        # Mental health related features
        df = self._create_mental_health_features(df)
        
        # Risk factor aggregations
        df = self._create_risk_aggregations(df)
        
        # Select final feature set
        feature_columns = self._select_final_features(df)
        
        # Keep only selected features + target
        final_columns = feature_columns + [self.target_column]
        df_final = df[final_columns].copy()
        
        self.feature_names = feature_columns
        self.df = df_final
        
        logger.info(f"Final feature set: {len(feature_columns)} features")
        logger.info(f"Final data shape: {df_final.shape}")
        
        return df_final
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        logger.info("Splitting data...")
        
        if self.df is None:
            self.engineer_features()
        
        df = self.df.dropna(subset=[self.target_column]).copy()
        
        # Features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG["test_size"],
            random_state=MODEL_CONFIG["random_state"],
            stratify=y
        )
        
        # Second split: train vs validation
        val_size = MODEL_CONFIG["validation_size"] / (1 - MODEL_CONFIG["test_size"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=MODEL_CONFIG["random_state"],
            stratify=y_temp
        )
        
        # Create DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Save splits
        train_df.to_csv(PROCESSED_DATA_DIR / DATA_CONFIG["train_file"], index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / DATA_CONFIG["validation_file"], index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / DATA_CONFIG["test_file"], index=False)
        
        logger.info(f"Train set: {train_df.shape}")
        logger.info(f"Validation set: {val_df.shape}")
        logger.info(f"Test set: {test_df.shape}")
        
        return train_df, val_df, test_df
    
    def preprocess_for_ml(self, train_df: pd.DataFrame, 
                         val_df: pd.DataFrame, 
                         test_df: pd.DataFrame) -> Dict[str, Any]:
        """Preprocess data for machine learning models"""
        logger.info("Preprocessing for ML...")
        
        # Separate features and targets
        X_train = train_df.drop(columns=[self.target_column])
        y_train = train_df[self.target_column]
        
        X_val = val_df.drop(columns=[self.target_column])
        y_val = val_df[self.target_column]
        
        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column]
        
        # Handle categorical and numerical features separately
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Categorical features: {len(categorical_features)}")
        logger.info(f"Numerical features: {len(numerical_features)}")
        
        # Process categorical features
        if categorical_features:
            X_train, X_val, X_test, encoders = self._encode_categorical_features(
                X_train, X_val, X_test, categorical_features
            )
        else:
            encoders = {}
        
        # Process numerical features
        if numerical_features:
            X_train, X_val, X_test, scalers = self._scale_numerical_features(
                X_train, X_val, X_test, numerical_features
            )
        else:
            scalers = {}
        
        # Final data package
        ml_data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist(),
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'encoders': encoders,
            'scalers': scalers
        }
        
        return ml_data
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete data processing pipeline"""
        logger.info("Running full data processing pipeline...")
        
        # Step 1: Load and clean data
        self.load_data()
        self.clean_data()
        
        # Step 2: Create target variable
        self.create_target_variable()
        
        # Step 3: Engineer features
        self.engineer_features()
        
        # Step 4: Split data
        train_df, val_df, test_df = self.split_data()
        
        # Step 5: Preprocess for ML
        ml_data = self.preprocess_for_ml(train_df, val_df, test_df)
        
        # Save processed data
        self.df.to_csv(self.processed_path, index=False)
        
        logger.info("Data processing pipeline completed successfully!")
        
        return ml_data
    
    # Helper methods
    def _clean_column_name(self, col_name: str) -> str:
        """Clean column names for consistency"""
        # Remove special characters and normalize
        cleaned = re.sub(r'[^\w\s]', '', str(col_name))
        cleaned = re.sub(r'\s+', '_', cleaned.strip())
        return cleaned.lower()
    
    def _clean_age_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean age column"""
        age_columns = [col for col in df.columns if 'age' in col.lower()]
        
        if not age_columns:
            return df
        
        age_col = age_columns[0]
        
        # Convert to numeric
        df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
        
        # Remove implausible ages
        df.loc[(df[age_col] < 16) | (df[age_col] > 100), age_col] = np.nan
        
        # Rename to standard name
        df = df.rename(columns={age_col: 'age'})
        
        return df
    
    def _clean_gender_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize gender column"""
        gender_columns = [col for col in df.columns if 'gender' in col.lower()]
        
        if not gender_columns:
            return df
        
        gender_col = gender_columns[0]
        
        # Normalize gender responses
        df[gender_col] = df[gender_col].astype(str).str.strip().str.lower()
        
        def normalize_gender(gender_str):
            if pd.isna(gender_str) or gender_str in ['nan', 'na', '']:
                return np.nan
            
            gender_str = str(gender_str).lower().strip()
            
            # Male patterns
            if any(pattern in gender_str for pattern in ['male', 'm', 'man']):
                if 'female' in gender_str or 'woman' in gender_str:
                    return 'Other'
                return 'Male'
            
            # Female patterns
            if any(pattern in gender_str for pattern in ['female', 'f', 'woman']):
                return 'Female'
            
            # Non-binary patterns
            if any(pattern in gender_str for pattern in 
                  ['non-binary', 'nonbinary', 'non binary', 'nb', 'enby']):
                return 'Non-binary'
            
            # Other/prefer not to say
            if any(pattern in gender_str for pattern in 
                  ['other', 'prefer', 'queer', 'fluid']):
                return 'Other'
            
            return 'Other'
        
        df[gender_col] = df[gender_col].apply(normalize_gender)
        df = df.rename(columns={gender_col: 'gender'})
        
        return df
    
    def _clean_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean categorical columns"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['gender']:  # Already handled
                # Normalize categorical responses
                df[col] = df[col].astype(str).str.strip()
                
                # Handle common yes/no responses
                df[col] = df[col].apply(self._normalize_yes_no_response_flexible)
        
        return df
    
    def _normalize_yes_no_response(self, response) -> int:
        """Normalize yes/no responses to binary (strict version for target)"""
        if pd.isna(response):
            return np.nan
        
        response_str = str(response).lower().strip()
        
        if response_str in ['yes', 'y', 'true', '1', '1.0']:
            return 1
        elif response_str in ['no', 'n', 'false', '0', '0.0']:
            return 0
        else:
            return np.nan
    
    def _normalize_yes_no_response_flexible(self, response):
        """Normalize yes/no responses (flexible version for features)"""
        if pd.isna(response) or str(response).lower() in ['nan', 'na', '']:
            return np.nan
        
        response_str = str(response).lower().strip()
        
        if response_str in ['yes', 'y', 'true', '1', '1.0']:
            return 'Yes'
        elif response_str in ['no', 'n', 'false', '0', '0.0']:
            return 'No'
        elif 'maybe' in response_str or 'not sure' in response_str:
            return 'Maybe'
        elif 'don' in response_str and 'know' in response_str:
            return "Don't know"
        else:
            return str(response).title()
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-based features"""
        if 'age' not in df.columns:
            return df
        
        # Age bins
        age_bins = FEATURE_CONFIG["age_bins"]
        age_labels = FEATURE_CONFIG["age_labels"]
        
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        
        # Age-related flags
        df['is_young_adult'] = (df['age'] <= 25).astype(int)
        df['is_senior'] = (df['age'] >= 50).astype(int)
        
        return df
    
    def _create_workplace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create workplace-related features"""
        workplace_keywords = ['employee', 'company', 'work', 'employer', 'tech', 'remote']
        workplace_cols = [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in workplace_keywords)]
        
        logger.info(f"Found workplace columns: {workplace_cols}")
        
        # Company size normalization
        size_cols = [col for col in df.columns if 'employee' in col.lower()]
        if size_cols:
            size_col = size_cols[0]
            df = df.rename(columns={size_col: 'company_size'})
        
        return df
    
    def _create_mental_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create mental health related features"""
        mh_keywords = ['mental', 'health', 'treatment', 'therapy', 'family', 'history']
        mh_cols = [col for col in df.columns 
                  if any(keyword in col.lower() for keyword in mh_keywords)]
        
        logger.info(f"Found mental health columns: {mh_cols}")
        
        # Family history normalization
        family_cols = [col for col in df.columns 
                      if 'family' in col.lower() and 'history' in col.lower()]
        if family_cols:
            family_col = family_cols[0]
            df['family_history'] = df[family_col].apply(self._normalize_yes_no_response)
            df = df.rename(columns={family_col: 'family_history_raw'})
        
        return df
    
    def _create_risk_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated risk features"""
        # Support availability score
        support_cols = [col for col in df.columns 
                       if any(keyword in col.lower() 
                             for keyword in ['help', 'support', 'resource', 'benefit'])]
        
        if len(support_cols) >= 2:
            def score_response(x):
                return 1 if str(x).lower() in ['yes', 'y'] else 0
            support_responses = df[support_cols].apply(lambda col: col.apply(score_response))
            df['support_availability_score'] = support_responses.sum(axis=1)
        
        # Workplace comfort score
        comfort_cols = [col for col in df.columns 
                       if 'comfortable' in col.lower() or 'discuss' in col.lower()]
        
        if len(comfort_cols) >= 1:
            def score_response(x):
                return 1 if str(x).lower() in ['yes', 'y'] else 0
            comfort_responses = df[comfort_cols].apply(lambda col: col.apply(score_response))
            df['workplace_comfort_score'] = comfort_responses.sum(axis=1)
        
        return df
    
    def _select_final_features(self, df: pd.DataFrame) -> List[str]:
        """Select final feature set for modeling"""
        # Exclude non-predictive columns
        exclude_patterns = [
            'comment', 'timestamp', 'token', 'completed', 'submit',
            self.target_column, '_raw'
        ]
        
        candidate_features = []
        for col in df.columns:
            if not any(pattern in col.lower() for pattern in exclude_patterns):
                # Check if column has sufficient non-null values
                if df[col].notna().sum() / len(df) >= 0.1:  # At least 10% non-null
                    candidate_features.append(col)
        
        logger.info(f"Selected {len(candidate_features)} candidate features")
        return candidate_features[:FEATURE_CONFIG["feature_selection_k"]]
    
    def _encode_categorical_features(self, X_train, X_val, X_test, categorical_features):
        """Encode categorical features"""
        encoders = {}
        
        for feature in categorical_features:
            # Use Label Encoding for simplicity (LightGBM handles categorical well)
            encoder = LabelEncoder()
            
            # Fit on training data
            encoder.fit(X_train[feature].fillna('Unknown'))
            
            # Transform all sets
            X_train[feature] = encoder.transform(X_train[feature].fillna('Unknown'))
            
            # Handle unseen categories in validation/test
            X_val[feature] = X_val[feature].fillna('Unknown')
            X_val[feature] = X_val[feature].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ 
                else encoder.transform(['Unknown'])[0]
            )
            
            X_test[feature] = X_test[feature].fillna('Unknown')
            X_test[feature] = X_test[feature].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ 
                else encoder.transform(['Unknown'])[0]
            )
            
            encoders[feature] = encoder
        
        return X_train, X_val, X_test, encoders
    
    def _scale_numerical_features(self, X_train, X_val, X_test, numerical_features):
        """Scale numerical features"""
        scalers = {}
        
        for feature in numerical_features:
            # Check if feature has any non-null values
            if X_train[feature].notna().sum() == 0:
                logger.warning(f"Feature {feature} has no non-null values, dropping")
                X_train = X_train.drop(columns=[feature])
                X_val = X_val.drop(columns=[feature])
                X_test = X_test.drop(columns=[feature])
                continue
                
            scaler = StandardScaler()
            imputer = SimpleImputer(strategy='median')
            
            # Fit and transform training data
            train_values = imputer.fit_transform(X_train[[feature]]).flatten()
            X_train.loc[:, feature] = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
            
            # Transform validation and test data
            val_values = imputer.transform(X_val[[feature]]).flatten()
            X_val.loc[:, feature] = scaler.transform(val_values.reshape(-1, 1)).flatten()
            
            test_values = imputer.transform(X_test[[feature]]).flatten()
            X_test.loc[:, feature] = scaler.transform(test_values.reshape(-1, 1)).flatten()
            
            scalers[feature] = {'scaler': scaler, 'imputer': imputer}
        
        return X_train, X_val, X_test, scalers


def main():
    """Main execution function"""
    processor = OSMIDataProcessor()
    ml_data = processor.run_full_pipeline()
    
    print("\n=== Data Processing Summary ===")
    print(f"Training samples: {len(ml_data['X_train'])}")
    print(f"Validation samples: {len(ml_data['X_val'])}")
    print(f"Test samples: {len(ml_data['X_test'])}")
    print(f"Number of features: {len(ml_data['feature_names'])}")
    print(f"Target distribution (train):")
    print(ml_data['y_train'].value_counts(normalize=True))
    
    return ml_data


if __name__ == "__main__":
    main()
