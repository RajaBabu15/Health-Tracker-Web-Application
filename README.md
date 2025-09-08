---
title: Mental Health Risk Predictor
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
short_description: Predict likelihood of seeking mental health treatment based on workplace factors
---

# 🧠 Mental Health Risk Predictor

An AI-powered tool to predict the likelihood of seeking mental health treatment based on workplace and personal factors using data from the Open Sourcing Mental Illness (OSMI) survey.

## 🎯 Features

- **Interactive Assessment**: Comprehensive form covering workplace mental health factors
- **Real-time ML Predictions**: Powered by optimized LightGBM model
- **Risk Visualization**: Professional gauge chart showing prediction confidence
- **Personalized Recommendations**: Tailored advice based on risk level
- **Validated Test Cases**: Color-coded examples from real dataset for model verification

## 📊 Model Performance

- **ROC AUC**: 0.798 (Excellent)
- **Accuracy**: 77%
- **Precision**: 80.95%
- **Calibration**: Excellent (Brier score: 0.167)

## 🤖 Technical Details

- **Algorithm**: LightGBM with probability calibration
- **Optimization**: Hyperparameter tuning with Optuna (100 trials)
- **Features**: 19 engineered features from workplace mental health survey
- **Data**: 826 clean samples from OSMI Mental Health in Tech Survey

## 🧪 Test Cases

The app includes verified test cases:
- **🔴 Red Cases**: Individuals who actually sought treatment
- **🟢 Green Cases**: Individuals who did not seek treatment

## 🚀 Usage

1. Fill out the assessment form with your workplace and personal information
2. Click "Predict Risk" to get your mental health treatment-seeking likelihood
3. View personalized recommendations based on your risk level
4. Try the test cases to see how the model performs on known outcomes

## ⚠️ Disclaimer

This tool is for informational purposes only and should not replace professional medical advice. If you're experiencing mental health issues, please consult with a qualified healthcare provider.

## 🔬 Research Background

Built using the Open Sourcing Mental Illness (OSMI) dataset, which surveys mental health attitudes and experiences in the tech workplace. The model helps identify factors that influence mental health treatment-seeking behavior.

---

*Developed with ❤️ for mental health awareness in the workplace*
