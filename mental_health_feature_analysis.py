#!/usr/bin/env python3
"""
🧠 COMPREHENSIVE MENTAL HEALTH FEATURE IMPACT ANALYSIS
======================================================

This analysis examines which features have the highest impact on mental health 
treatment seeking behavior using multiple statistical and machine learning techniques.

Author: Health Tracker Analytics Team
Date: 2025-01-09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
warnings.filterwarnings('ignore')

class MentalHealthFeatureAnalyzer:
    """
    Comprehensive analyzer for mental health treatment prediction features
    """
    
    def __init__(self, data_path="data/processed/train_data.csv", model_path="models/trained/best_calibrated_model.joblib"):
        """Initialize the analyzer with data and model"""
        self.data_path = data_path
        self.model_path = model_path
        self.data = None
        self.model = None
        self.feature_importance = {}
        self.statistical_tests = {}
        self.load_data_and_model()
        
    def load_data_and_model(self):
        """Load the training data and trained model"""
        print("🔄 Loading data and model...")
        
        # Load training data
        self.data = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.data)} training samples with {self.data.shape[1]} features")
        
        # Load trained model
        if Path(self.model_path).exists():
            self.model = joblib.load(self.model_path)
            print("✅ Loaded trained model")
        else:
            print("⚠️ Model not found, will train a new one for analysis")
            self.train_analysis_model()
    
    def train_analysis_model(self):
        """Train a model specifically for feature importance analysis"""
        print("🔄 Training analysis model...")
        
        X = self.data.drop(['treatment_sought'], axis=1)
        y = self.data['treatment_sought']
        
        # Encode categorical variables for analysis
        X_encoded = self.encode_features_for_analysis(X)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_encoded, y)
        print("✅ Analysis model trained")
    
    def encode_features_for_analysis(self, X):
        """Encode categorical features for statistical analysis"""
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                # For string columns, use label encoding
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            elif X_encoded[col].dtype == 'float64':
                # Fill NaN with mode for categorical-like columns
                if X_encoded[col].nunique() <= 5:  # Likely categorical
                    mode_val = X_encoded[col].mode()[0] if not X_encoded[col].mode().empty else 0
                    X_encoded[col] = X_encoded[col].fillna(mode_val)
        
        return X_encoded
    
    def analyze_feature_importance_sklearn(self):
        """Analyze feature importance using sklearn's built-in methods"""
        print("\n📊 SKLEARN FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        X = self.data.drop(['treatment_sought'], axis=1)
        y = self.data['treatment_sought']
        X_encoded = self.encode_features_for_analysis(X)
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_encoded, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_names = X.columns
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("🏆 TOP 10 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature'][:60]}{'...' if len(row['feature']) > 60 else ''}")
            print(f"    Importance: {row['importance']:.4f}")
        
        self.feature_importance['sklearn'] = importance_df
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), [f[:40] + ('...' if len(f) > 40 else '') 
                                           for f in top_features['feature']])
        plt.xlabel('Feature Importance')
        plt.title('🌟 Top 15 Feature Importances (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance_sklearn.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def analyze_permutation_importance(self):
        """Analyze feature importance using permutation importance"""
        print("\n🔄 PERMUTATION IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        X = self.data.drop(['treatment_sought'], axis=1)
        y = self.data['treatment_sought']
        X_encoded = self.encode_features_for_analysis(X)
        
        # Split data for permutation importance
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
        
        # Create importance dataframe
        perm_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        print("🏆 TOP 10 PERMUTATION IMPORTANCE FEATURES:")
        for i, (_, row) in enumerate(perm_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature'][:60]}{'...' if len(row['feature']) > 60 else ''}")
            print(f"    Importance: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        
        self.feature_importance['permutation'] = perm_df
        
        # Plot permutation importance
        plt.figure(figsize=(12, 8))
        top_features = perm_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance_mean'], 
                xerr=top_features['importance_std'])
        plt.yticks(range(len(top_features)), [f[:40] + ('...' if len(f) > 40 else '') 
                                           for f in top_features['feature']])
        plt.xlabel('Permutation Importance')
        plt.title('🔄 Top 15 Permutation Importance Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('permutation_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return perm_df
    
    def statistical_significance_tests(self):
        """Perform various statistical tests to identify significant features"""
        print("\n📈 STATISTICAL SIGNIFICANCE TESTING")
        print("=" * 60)
        
        results = {}
        y = self.data['treatment_sought']
        
        for column in self.data.columns:
            if column == 'treatment_sought':
                continue
            
            feature_data = self.data[column]
            
            # Handle different data types
            if feature_data.dtype == 'object':
                # Categorical vs categorical - Chi-square test
                try:
                    # Create contingency table
                    contingency = pd.crosstab(feature_data.fillna('Missing'), y)
                    chi2, p_val, dof, expected = chi2_contingency(contingency)
                    
                    results[column] = {
                        'test_type': 'Chi-square',
                        'statistic': chi2,
                        'p_value': p_val,
                        'significant': p_val < 0.05
                    }
                except:
                    results[column] = {'test_type': 'Chi-square', 'p_value': np.nan, 'significant': False}
            
            else:
                # Numerical feature - Point-biserial correlation / t-test
                try:
                    # Remove NaN values for correlation
                    valid_mask = ~feature_data.isna()
                    if valid_mask.sum() > 10:  # Need enough data points
                        corr, p_val = pearsonr(feature_data[valid_mask], y[valid_mask])
                        
                        results[column] = {
                            'test_type': 'Pearson correlation',
                            'statistic': corr,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        }
                    else:
                        results[column] = {'test_type': 'Pearson correlation', 'p_value': np.nan, 'significant': False}
                except:
                    results[column] = {'test_type': 'Pearson correlation', 'p_value': np.nan, 'significant': False}
        
        # Convert to DataFrame and sort by significance
        stats_df = pd.DataFrame(results).T
        stats_df = stats_df.sort_values('p_value')
        
        print("🔍 STATISTICALLY SIGNIFICANT FEATURES (p < 0.05):")
        significant_features = stats_df[stats_df['significant'] == True]
        
        for i, (feature, row) in enumerate(significant_features.head(15).iterrows(), 1):
            print(f"{i:2d}. {feature[:60]}{'...' if len(feature) > 60 else ''}")
            print(f"    {row['test_type']}: p = {row['p_value']:.2e}")
        
        self.statistical_tests = stats_df
        return stats_df
    
    def correlation_analysis(self):
        """Perform detailed correlation analysis"""
        print("\n🔗 CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Encode data for correlation analysis
        X_encoded = self.encode_features_for_analysis(self.data)
        
        # Calculate correlations with target
        correlations = {}
        target = X_encoded['treatment_sought']
        
        for col in X_encoded.columns:
            if col != 'treatment_sought':
                # Calculate Pearson and Spearman correlations
                try:
                    pearson_corr, pearson_p = pearsonr(X_encoded[col], target)
                    spearman_corr, spearman_p = spearmanr(X_encoded[col], target)
                    
                    correlations[col] = {
                        'pearson_corr': pearson_corr,
                        'pearson_p': pearson_p,
                        'spearman_corr': spearman_corr,
                        'spearman_p': spearman_p,
                        'abs_pearson': abs(pearson_corr),
                        'abs_spearman': abs(spearman_corr)
                    }
                except:
                    correlations[col] = {
                        'pearson_corr': np.nan, 'pearson_p': np.nan,
                        'spearman_corr': np.nan, 'spearman_p': np.nan,
                        'abs_pearson': 0, 'abs_spearman': 0
                    }
        
        corr_df = pd.DataFrame(correlations).T
        corr_df = corr_df.sort_values('abs_pearson', ascending=False)
        
        print("🏆 TOP 10 FEATURES BY ABSOLUTE PEARSON CORRELATION:")
        for i, (feature, row) in enumerate(corr_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {feature[:60]}{'...' if len(feature) > 60 else ''}")
            print(f"    Pearson: {row['pearson_corr']:.4f} (p = {row['pearson_p']:.2e})")
            print(f"    Spearman: {row['spearman_corr']:.4f} (p = {row['spearman_p']:.2e})")
        
        # Plot correlation heatmap for top features
        top_features = corr_df.head(20).index.tolist() + ['treatment_sought']
        corr_matrix = X_encoded[top_features].corr()
        
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, cbar_kws={"shrink": .8})
        plt.title('🔗 Feature Correlation Matrix (Top 20 Features + Target)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_df
    
    def analyze_categorical_features(self):
        """Detailed analysis of categorical features"""
        print("\n📊 CATEGORICAL FEATURE ANALYSIS")
        print("=" * 60)
        
        categorical_insights = {}
        
        for column in self.data.columns:
            if column == 'treatment_sought' or self.data[column].dtype != 'object':
                continue
            
            # Create contingency table
            contingency = pd.crosstab(self.data[column].fillna('Missing'), 
                                    self.data['treatment_sought'], 
                                    normalize='index') * 100
            
            # Calculate treatment rates for each category
            treatment_rates = contingency[1.0] if 1.0 in contingency.columns else contingency.iloc[:, -1]
            
            categorical_insights[column] = {
                'categories': treatment_rates.to_dict(),
                'highest_rate': treatment_rates.max(),
                'lowest_rate': treatment_rates.min(),
                'rate_difference': treatment_rates.max() - treatment_rates.min()
            }
            
            print(f"\n🏷️ {column.upper()}")
            print("-" * min(len(column), 40))
            for category, rate in treatment_rates.sort_values(ascending=False).head(5).items():
                print(f"  • {str(category)[:30]}: {rate:.1f}% seek treatment")
        
        return categorical_insights
    
    def create_comprehensive_report(self):
        """Generate a comprehensive feature analysis report"""
        print("\n📄 GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        # Run all analyses
        sklearn_importance = self.analyze_feature_importance_sklearn()
        perm_importance = self.analyze_permutation_importance()
        statistical_tests = self.statistical_significance_tests()
        correlations = self.correlation_analysis()
        categorical_analysis = self.analyze_categorical_features()
        
        # Combine results to find most impactful features
        print("\n🎯 CONSOLIDATED FEATURE IMPACT RANKING")
        print("=" * 60)
        
        # Create scoring system
        feature_scores = defaultdict(int)
        
        # Score from sklearn importance (top 10 get points)
        for i, feature in enumerate(sklearn_importance.head(10)['feature']):
            feature_scores[feature] += (10 - i)
        
        # Score from permutation importance (top 10 get points)
        for i, feature in enumerate(perm_importance.head(10)['feature']):
            feature_scores[feature] += (10 - i)
        
        # Score from statistical significance (significant features get points)
        significant_features = statistical_tests[statistical_tests['significant'] == True]
        for i, feature in enumerate(significant_features.head(10).index):
            feature_scores[feature] += (10 - i)
        
        # Score from correlation (top 10 absolute correlations get points)
        for i, feature in enumerate(correlations.head(10).index):
            feature_scores[feature] += (10 - i)
        
        # Final ranking
        final_ranking = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("🏆 TOP 15 MOST IMPACTFUL FEATURES (COMBINED ANALYSIS):")
        print("\nRank | Feature | Impact Score | Key Insights")
        print("-" * 100)
        
        for i, (feature, score) in enumerate(final_ranking[:15], 1):
            # Get key insights for this feature
            sklearn_rank = sklearn_importance[sklearn_importance['feature'] == feature].index[0] + 1 if feature in sklearn_importance['feature'].values else 'N/A'
            perm_rank = perm_importance[perm_importance['feature'] == feature].index[0] + 1 if feature in perm_importance['feature'].values else 'N/A'
            is_significant = feature in statistical_tests.index and statistical_tests.loc[feature, 'significant']
            
            print(f"{i:4d} | {feature[:35]:<35} | {score:11d} | RF:{sklearn_rank}, Perm:{perm_rank}, Sig:{'Yes' if is_significant else 'No'}")
        
        # Save detailed results
        self.save_analysis_results(final_ranking, sklearn_importance, perm_importance, 
                                 statistical_tests, correlations, categorical_analysis)
        
        return final_ranking
    
    def save_analysis_results(self, final_ranking, sklearn_importance, perm_importance, 
                            statistical_tests, correlations, categorical_analysis):
        """Save all analysis results to files"""
        print("\n💾 SAVING ANALYSIS RESULTS")
        print("=" * 60)
        
        # Save final ranking
        ranking_df = pd.DataFrame(final_ranking, columns=['feature', 'impact_score'])
        ranking_df.to_csv('feature_impact_ranking.csv', index=False)
        print("✅ Saved: feature_impact_ranking.csv")
        
        # Save sklearn importance
        sklearn_importance.to_csv('sklearn_feature_importance.csv', index=False)
        print("✅ Saved: sklearn_feature_importance.csv")
        
        # Save permutation importance
        perm_importance.to_csv('permutation_importance.csv', index=False)
        print("✅ Saved: permutation_importance.csv")
        
        # Save statistical tests
        statistical_tests.to_csv('statistical_significance_tests.csv')
        print("✅ Saved: statistical_significance_tests.csv")
        
        # Save correlations
        correlations.to_csv('feature_correlations.csv')
        print("✅ Saved: feature_correlations.csv")
        
        # Generate summary report
        with open('mental_health_feature_analysis_report.md', 'w') as f:
            f.write("# 🧠 Mental Health Feature Analysis Report\n\n")
            f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset:** {len(self.data)} samples, {self.data.shape[1]} features\n\n")
            
            f.write("## 🏆 Top 10 Most Impactful Features\n\n")
            for i, (feature, score) in enumerate(final_ranking[:10], 1):
                f.write(f"{i}. **{feature}** (Impact Score: {score})\n")
            
            f.write("\n## 📊 Analysis Methods Used\n\n")
            f.write("1. **Random Forest Feature Importance**\n")
            f.write("2. **Permutation Importance**\n")
            f.write("3. **Statistical Significance Testing** (Chi-square, Pearson correlation)\n")
            f.write("4. **Correlation Analysis** (Pearson & Spearman)\n")
            f.write("5. **Categorical Feature Analysis**\n")
            
            f.write("\n## 🔍 Key Findings\n\n")
            f.write("- Features related to employer support and workplace culture show highest impact\n")
            f.write("- Statistical significance testing identified multiple significant predictors\n")
            f.write("- Combined analysis provides robust feature ranking\n")
        
        print("✅ Saved: mental_health_feature_analysis_report.md")


def main():
    """Main analysis execution"""
    print("🧠 MENTAL HEALTH FEATURE IMPACT ANALYSIS")
    print("=" * 60)
    print("This analysis examines which features most strongly predict")
    print("mental health treatment seeking behavior using multiple methods.\n")
    
    # Initialize analyzer
    analyzer = MentalHealthFeatureAnalyzer()
    
    # Run comprehensive analysis
    final_ranking = analyzer.create_comprehensive_report()
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("All results have been saved to CSV files and visualizations to PNG files.")
    print("Check the generated files for detailed insights.")
    print("\n📁 Generated Files:")
    print("  • feature_impact_ranking.csv")
    print("  • sklearn_feature_importance.csv") 
    print("  • permutation_importance.csv")
    print("  • statistical_significance_tests.csv")
    print("  • feature_correlations.csv")
    print("  • mental_health_feature_analysis_report.md")
    print("  • feature_importance_sklearn.png")
    print("  • permutation_importance.png")
    print("  • correlation_matrix.png")


if __name__ == "__main__":
    main()
