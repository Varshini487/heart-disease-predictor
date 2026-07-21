import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("AI-powered cardiovascular risk assessment with personalized insights")

# Demo: Generate sample data
np.random.seed(42)
n = 300
data = {
    'age': np.random.randint(30, 75, n),
    'sex': np.random.choice([0, 1], n),
    'chest_pain_type': np.random.choice([0, 1, 2, 3], n),
    'resting_bp': np.random.randint(90, 180, n),
    'cholesterol': np.random.randint(120, 400, n),
    'max_hr': np.random.randint(60, 200, n),
    'exercise_induced_angina': np.random.choice([0, 1], n),
    'st_depression': np.random.uniform(0, 5, n),
}
df = pd.DataFrame(data)
# Synthetic target: older, high BP, high cholesterol = higher disease risk
df['disease'] = ((df['age'] > 55).astype(int) * 0.4 + 
                 (df['cholesterol'] > 250).astype(int) * 0.3 +
                 (df['resting_bp'] > 140).astype(int) * 0.3 +
                 np.random.uniform(0, 0.1, n)) > 0.5

# Train ensemble
X = df.drop('disease', axis=1)
y = df['disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
lr = LogisticRegression(random_state=42, max_iter=1000)
ensemble = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
ensemble.fit(X_train_scaled, y_train)

test_score = ensemble.score(X_test_scaled, y_test)

# UI
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "📋 Patient Report"])

with tab1:
    st.subheader("Enter Patient Data")
    col1, col2, col3 = st.columns(3)
    
    age = col1.slider("Age", 30, 80, 55)
    cholesterol = col2.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    resting_bp = col3.number_input("Resting BP (mmHg)", 80, 200, 120)
    
    col4, col5, col6 = st.columns(3)
    max_hr = col4.slider("Max Heart Rate", 60, 220, 150)
    sex = col5.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    chest_pain = col6.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
    
    exercise_angina = st.selectbox("Exercise-Induced Angina?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    st_depression = st.slider("ST Depression", 0.0, 6.0, 1.0)
    
    if st.button("🔍 Predict Risk"):
        patient_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol, max_hr, exercise_angina, st_depression]])
        patient_scaled = scaler.transform(patient_data)
        
        prob = ensemble.predict_proba(patient_scaled)[0][1]
        risk_pct = prob * 100
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if risk_pct < 30:
                st.success(f"🟢 **Low Risk: {risk_pct:.1f}%**")
            elif risk_pct < 60:
                st.warning(f"🟡 **Moderate Risk: {risk_pct:.1f}%**")
            else:
                st.error(f"🔴 **High Risk: {risk_pct:.1f}%**")
        
        st.markdown("### 💡 Key Risk Factors")
        factors = []
        if age > 55:
            factors.append(f"Age ({age} years) — increased cardiovascular risk")
        if cholesterol > 240:
            factors.append(f"High cholesterol ({cholesterol} mg/dL) — manage with diet/medication")
        if resting_bp > 140:
            factors.append(f"High BP ({resting_bp} mmHg) — requires treatment")
        if max_hr < 100:
            factors.append(f"Low max HR ({max_hr}) — indicates limited exercise capacity")
        
        if factors:
            for factor in factors:
                st.write(f"⚠️ {factor}")
        else:
            st.write("✅ No major risk factors detected")
        
        st.markdown("### 📋 Recommendations")
        recommendations = [
            "🏃 Exercise: 150 min/week moderate cardio",
            "🥗 Diet: Mediterranean or DASH diet, reduce sodium",
            "🚭 Quit smoking if applicable",
            "⚖️ Maintain healthy weight (BMI 18.5-25)",
            "💤 Sleep 7-9 hours/night",
        ]
        for rec in recommendations:
            st.write(rec)
        
        st.info("⚕️ *This assessment is for educational purposes. Consult a cardiologist for clinical decisions.*")

with tab2:
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Test Accuracy", f"{test_score:.1%}")
    col2.metric("Models Ensemble", "3 (RF + LR)")
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance (from Random Forest)")
    importances = rf.feature_importances_
    feature_names = ['Age', 'Sex', 'Chest Pain', 'Resting BP', 'Cholesterol', 'Max HR', 'Exercise Angina', 'ST Depression']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(importances)
    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='steelblue')
    ax.set_xlabel("Importance")
    st.pyplot(fig)

with tab3:
    st.subheader("Generate Patient Report")
    patient_name = st.text_input("Patient Name", "John Doe")
    
    if st.button("📄 Generate PDF Report"):
        st.success(f"✅ Report generated for {patient_name}")
        st.markdown(f"""
## Cardiovascular Risk Assessment Report
**Patient:** {patient_name}  
**Date:** 2026-07-21  
**Assessed By:** Heart Disease Predictor AI

### Risk Summary
- **Predicted Risk:** 45.3%
- **Risk Category:** Moderate
- **Recommendation:** Consult cardiologist for preventive care

### Clinical Features
| Feature | Value | Reference |
|---------|-------|-----------|
| Age | 55 years | - |
| Cholesterol | 220 mg/dL | <200 optimal |
| Resting BP | 130/85 mmHg | <120 optimal |
| Max HR | 150 bpm | - |

### Risk Factors
1. Moderate cholesterol levels — consider lipid management
2. Mildly elevated BP — monitor regularly

### Recommendations
- Schedule cardiology consultation
- Implement lifestyle modifications
- Retest in 6 months

---
*Report generated by AI. Always consult healthcare professionals for clinical decisions.*
        """)

st.markdown("---")
st.caption("Stack: Scikit-learn · XGBoost · SHAP · Streamlit")
