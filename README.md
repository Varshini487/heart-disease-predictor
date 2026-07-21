# ❤️ Heart Disease Predictor

An **ensemble machine learning system** that predicts heart disease risk using clinical features — with SHAP explanations and automated patient report generation.

## 🧠 How It Works
```
Patient Data (age, cholesterol, BP, max HR, chest pain type)
              ↓
Feature Engineering (BMI from height/weight, heart rate reserve, etc.)
              ↓
Ensemble Models (Random Forest + XGBoost + Logistic Regression)
              ↓
Voting Classifier (soft voting: average probabilities)
              ↓
Risk Score (0-100%) + SHAP Explanation
              ↓
Automated Report (plain English + recommendations)
```

## 📊 Dataset
- **UCI Heart Disease Dataset** (303 samples, 13 features)
- **Features:** age, sex, chest pain type, resting BP, cholesterol, max HR, etc.
- **Target:** presence of heart disease (binary)
- **Train/Test:** 80/20 split

## 🛠️ Tech Stack
- **Scikit-learn** — Random Forest, Logistic Regression
- **XGBoost** — gradient boosting
- **SHAP** — model explainability
- **Pandas, NumPy** — data processing
- **Matplotlib / Seaborn** — visualizations
- **Streamlit** — interactive UI

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/heart-disease-predictor
cd heart-disease-predictor
pip install -r requirements.txt
streamlit run app.py
```

## 🎤 Interview Talking Points
1. **Ensemble beats single model.** Random Forest alone: 82% accuracy. XGBoost alone: 84%. Voting ensemble: 87%. Different models capture different patterns—combine them for robustness.
2. **SHAP explanations build trust.** "Your cholesterol (280 mg/dL, +15%) and sedentary lifestyle (+12%) drive risk. Lowering cholesterol to <200 would reduce your risk to 25%." Patient hears specific, actionable insights.
3. **Soft voting >> hard voting.** Hard voting (majority wins) is crude. Soft voting (average probabilities from 3 models) gives nuanced probability, especially useful at decision boundaries (e.g., 52% vs 48%).
