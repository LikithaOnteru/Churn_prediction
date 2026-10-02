import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from src.recommendations import generate_retention_recommendations

st.set_page_config(page_title="Customer Churn & Retention Intelligence", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/model.pkl')
    pipeline = joblib.load('models/pipeline.pkl')
    return model, pipeline

try:
    model, pipeline = load_artifacts()
except Exception as e:
    st.error("Error loading model artifacts. Please run `python src/train.py` first.")
    st.stop()

st.title("📊 Customer Churn Prediction & Retention Dashboard")
st.write("Input customer attributes to calculate real-time churn probability, view SHAP feature importances, and retrieve retention strategies.")

st.sidebar.header("Customer Profile Input")

# Dynamic inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", ["0", "1"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

input_dict = {
    'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
    'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple_lines,
    'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': backup,
    'DeviceProtection': device, 'TechSupport': tech, 'StreamingTV': tv,
    'StreamingMovies': movies, 'Contract': contract, 'PaperlessBilling': paperless,
    'PaymentMethod': payment, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_dict])

if st.button("Calculate Churn Risk", type="primary"):
    # Apply pipeline transformation
    input_trans = pipeline.transform(input_df)
    churn_proba = model.predict_proba(input_trans)[0][1]
    
    # 4-Tier Risk Segmentation
    if churn_proba < 0.30:
        risk_tier, color = "Low Risk", "green"
    elif churn_proba < 0.60:
        risk_tier, color = "Medium Risk", "orange"
    elif churn_proba < 0.80:
        risk_tier, color = "High Risk", "red"
    else:
        risk_tier, color = "Critical Risk", "darkred"

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric(label="Calculated Churn Probability", value=f"{churn_proba * 100:.1f}%")
        st.markdown(f"### Risk Tier: <span style='color:{color}'>{risk_tier}</span>", unsafe_allow_html=True)
        
    with col2:
        st.subheader("💡 Rule-Based Retention Recommendations")
        recs = generate_retention_recommendations(input_dict, churn_proba)
        for r in recs:
            st.write(r)

    st.markdown("---")
    st.subheader("🔍 Local SHAP Feature Contribution")
    
    # Local SHAP plot for the user's specific input
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(input_trans)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.bar(shap_vals[0], max_display=7, show=False)
    st.pyplot(fig)