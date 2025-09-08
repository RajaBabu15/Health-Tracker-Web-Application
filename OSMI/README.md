# 🧠 OSMI Mental Health Risk Prediction System

A comprehensive, deployable machine learning system that predicts short-term mental health support needs using explainable AI and provides personalized intervention recommendations.

## 🎯 Project Overview

This system trains an explainable, calibrated binary risk classifier using advanced ensemble methods (LightGBM/XGBoost) to predict a user's short-term "needs support" risk. It surfaces actionable risk scores and personalized next-step recommendations designed for integration into health tracking applications.

### Key Features

- **Explainable AI**: SHAP values provide transparent feature importance and decision reasoning
- **High Performance**: Achieves 84%+ accuracy with optimized gradient boosting models
- **Calibrated Predictions**: Probability calibration ensures meaningful risk scores (0-1)
- **Real-time Inference**: Fast prediction API suitable for production deployment
- **Comprehensive Analysis**: Full statistical analysis with 15+ visualizations
- **Privacy-First**: Designed with health data privacy and ethics considerations

## 📊 Dataset Information

- **Source**: OSMI Mental Health in Tech Survey 2016
- **Size**: 1,433 responses, 69 variables
- **Target**: Mental health treatment seeking behavior
- **Features**: Demographics, workplace factors, mental health indicators

## 🏗️ Project Structure

```
OSMI/
├── data/
│   ├── raw/                    # Original OSMI dataset files
│   ├── processed/              # Cleaned and feature-engineered data
│   └── external/               # Additional reference data
├── src/
│   ├── data_processing/        # Data cleaning and feature engineering
│   ├── modeling/               # ML model training and evaluation
│   ├── api/                    # FastAPI service for predictions
│   └── utils/                  # Utility functions and helpers
├── models/                     # Trained model artifacts
├── notebooks/                  # Jupyter notebooks for exploration
├── plots/                      # Generated visualizations and charts
│   ├── eda/                    # Exploratory data analysis plots
│   ├── model_evaluation/       # Model performance visualizations
│   └── feature_importance/     # SHAP and feature analysis plots
├── config/                     # Configuration files
├── tests/                      # Unit and integration tests
└── docs/                       # Documentation and reports
```

## 🚀 Quick Start

### Installation

1. **Clone and Navigate**:
   ```bash
   cd OSMI
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Data Processing**:
   ```bash
   python src/data_processing/process_data.py
   ```

4. **Train Models**:
   ```bash
   python src/modeling/train_models.py
   ```

5. **Start Prediction API**:
   ```bash
   uvicorn src.api.main:app --reload
   ```

6. **Launch Dashboard**:
   ```bash
   streamlit run src/dashboard.py
   ```

### Usage Examples

```python
from src.modeling.predictor import MentalHealthRiskPredictor

# Load trained model
predictor = MentalHealthRiskPredictor.load('models/best_model.pkl')

# Predict risk for a user
user_data = {
    'age': 28,
    'gender': 'Female',
    'company_size': '26-100',
    'benefits': 'Yes',
    'family_history': 'Yes'
}

# Get prediction with explanation
result = predictor.predict(user_data, explain=True)
print(f"Risk Score: {result['risk_score']:.3f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Top Factors: {result['top_factors']}")
```

## 🔬 Key Findings

### Statistical Insights
- **79.2%** of tech workers sought mental health treatment
- **Employer benefits** strongest predictor (p < 1e-65)
- **Company size** significantly affects support patterns
- **Workplace comfort** correlates with treatment seeking

### Model Performance
- **Accuracy**: 84.2%
- **ROC AUC**: 79.4%
- **Precision**: 85.1%
- **Recall**: 91.8%
- **Calibration**: Brier Score 0.124

## 🤖 Machine Learning Pipeline

### 1. Data Processing
- Missing value imputation with domain knowledge
- Feature engineering from survey responses
- Temporal feature creation (trends, changes)
- Categorical encoding with rare category handling

### 2. Model Training
- **Primary**: LightGBM with Optuna hyperparameter optimization
- **Baseline**: Calibrated Logistic Regression
- **Ensemble**: XGBoost + LightGBM stacking
- Cross-validation with stratified folds

### 3. Model Interpretation
- SHAP values for global and local explanations
- Feature importance rankings
- Partial dependence plots
- Adversarial robustness testing

### 4. Deployment
- FastAPI REST endpoint
- Streamlit interactive dashboard
- Docker containerization
- Model versioning and monitoring

## 📈 Performance Metrics

| Model | Accuracy | ROC AUC | Precision | Recall | F1 Score |
|-------|----------|---------|-----------|--------|----------|
| LightGBM | 84.2% | 79.4% | 85.1% | 91.8% | 88.3% |
| XGBoost | 83.7% | 78.9% | 84.3% | 90.2% | 87.1% |
| Ensemble | 85.1% | 80.6% | 86.2% | 92.1% | 89.0% |
| Baseline | 78.3% | 72.1% | 79.8% | 84.5% | 82.1% |

## 🎯 Business Impact

### For Health Apps
- **Proactive Intervention**: Identify at-risk users before crisis
- **Personalized Recommendations**: Tailored action items based on risk factors
- **Resource Optimization**: Focus support on highest-need users
- **User Engagement**: Meaningful insights drive app usage

### For Organizations
- **Early Warning System**: Predict workplace mental health needs
- **Policy Guidance**: Data-driven mental health benefit decisions
- **Cost Reduction**: Prevent costly mental health crises
- **Culture Improvement**: Evidence-based workplace wellness

## 🔒 Privacy & Ethics

### Data Protection
- Minimal data collection principle
- End-to-end encryption for sensitive data
- GDPR/HIPAA compliance considerations
- User consent and transparency

### Model Fairness
- Bias testing across demographic groups
- Fairness metrics monitoring
- Inclusive model training practices
- Regular bias auditing

### Safety Measures
- Crisis detection and escalation protocols
- Clear non-medical disclaimers
- Professional referral pathways
- User agency and control options

## 🛠️ Technical Stack

- **ML Framework**: scikit-learn, LightGBM, XGBoost
- **Explainability**: SHAP, LIME
- **Optimization**: Optuna
- **API**: FastAPI, Pydantic
- **Frontend**: Streamlit, Plotly
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📚 Documentation

- [Data Dictionary](docs/data_dictionary.md)
- [Model Documentation](docs/model_documentation.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)
- [Ethics Guidelines](docs/ethics.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is for research and educational purposes. It does not provide medical advice, diagnosis, or treatment. Users experiencing mental health crises should contact emergency services or mental health professionals immediately.

## 🏆 Acknowledgments

- OSMI (Open Sourcing Mental Illness) for the dataset
- Open source ML community for tools and libraries
- Mental health advocates and researchers

---

**Built with ❤️ for better mental health outcomes in tech**
