#!/usr/bin/env python3
"""
Quick exploration to identify data leakage issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv('data/processed/osmi_processed.csv')
print(f"Original shape: {df.shape}")

# Remove missing targets
df_clean = df.dropna(subset=['treatment_sought'])
print(f"After removing missing targets: {df_clean.shape}")

# Check for features with very high correlation with target
X = df_clean.drop(columns=['treatment_sought'])
y = df_clean['treatment_sought'].astype(int)

print(f"\nTarget distribution: {y.value_counts(normalize=True)}")

# Check each feature individually for perfect prediction
potential_leakage = []

for col in X.columns:
    # Skip columns with too many missing values
    if X[col].isnull().sum() / len(X) > 0.8:
        continue
    
    # For categorical columns, encode them
    if X[col].dtype == 'object':
        # Create a simple contingency table
        crosstab = pd.crosstab(X[col].fillna('Missing'), y, normalize='index')
        max_prob = crosstab.max(axis=1).max()
        if max_prob > 0.95:  # If any category predicts >95% of one class
            potential_leakage.append((col, max_prob))
            print(f"\nPotential leakage in {col}:")
            print(crosstab)
    
    # For numeric columns
    elif X[col].dtype in ['int64', 'float64']:
        # Check if it's actually binary/categorical encoded as numeric
        unique_vals = X[col].dropna().unique()
        if len(unique_vals) <= 5:
            crosstab = pd.crosstab(X[col].fillna(-999), y, normalize='index')
            max_prob = crosstab.max(axis=1).max()
            if max_prob > 0.95:
                potential_leakage.append((col, max_prob))
                print(f"\nPotential leakage in {col}:")
                print(crosstab)

print(f"\n{'='*60}")
print("POTENTIAL DATA LEAKAGE FEATURES:")
for col, prob in potential_leakage:
    print(f"  {col}: max prediction probability = {prob:.3f}")

# Test with a simple model on all features vs without suspicious features
print(f"\n{'='*60}")
print("MODEL PERFORMANCE COMPARISON:")

# Prepare data for modeling
X_encoded = X.copy()
le_dict = {}

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].fillna('Missing'))
        le_dict[col] = le

# Fill numeric missing values
X_encoded = X_encoded.fillna(-999)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Test 1: All features
lr_all = LogisticRegression(max_iter=1000, random_state=42)
lr_all.fit(X_train, y_train)
y_pred_all = lr_all.predict_proba(X_test)[:, 1]
auc_all = roc_auc_score(y_test, y_pred_all)

print(f"AUC with all features: {auc_all:.4f}")

# Test 2: Remove potential leakage features
suspicious_cols = [col for col, _ in potential_leakage]
if suspicious_cols:
    X_train_clean = X_train.drop(columns=suspicious_cols)
    X_test_clean = X_test.drop(columns=suspicious_cols)
    
    lr_clean = LogisticRegression(max_iter=1000, random_state=42)
    lr_clean.fit(X_train_clean, y_train)
    y_pred_clean = lr_clean.predict_proba(X_test_clean)[:, 1]
    auc_clean = roc_auc_score(y_test, y_pred_clean)
    
    print(f"AUC without suspicious features: {auc_clean:.4f}")
    print(f"Removed features: {suspicious_cols}")
else:
    print("No suspicious features identified")

print(f"\n{'='*60}")
print("FEATURE STATISTICS:")
print(f"Total features: {len(X.columns)}")
print(f"Features with >80% missing: {sum(X[col].isnull().sum() / len(X) > 0.8 for col in X.columns)}")
print(f"Numeric features: {sum(X[col].dtype in ['int64', 'float64'] for col in X.columns)}")
print(f"Categorical features: {sum(X[col].dtype == 'object' for col in X.columns)}")
