import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page layout
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("Customer Churn Prediction Dashboard")
st.write("Enter customer attributes below to calculate real-time churn risk.")

# Load trained model and column structure
@st.cache_resource
def load_assets():
    model = joblib.load('xgb_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

try:
    model, model_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Form layout for inputs
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=780.0)
        
    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
    submit_button = st.form_submit_button("Predict Churn Risk")

if submit_button:
    # 1. Create a baseline dictionary with zero for all model features
    input_dict = {col: 0 for col in model_columns}
    
    # 2. Populate numeric features
    if 'tenure' in input_dict: input_dict['tenure'] = tenure
    if 'MonthlyCharges' in input_dict: input_dict['MonthlyCharges'] = monthly_charges
    if 'TotalCharges' in input_dict: input_dict['TotalCharges'] = total_charges
    
    # 3. Populate categorical dummy variables matching pandas get_dummies naming
    contract_col = f"Contract_{contract}"
    if contract_col in input_dict:
        input_dict[contract_col] = 1
        
    internet_col = f"InternetService_{internet_service}"
    if internet_col in input_dict:
        input_dict[internet_col] = 1
        
    payment_col = f"PaymentMethod_{payment_method}"
    if payment_col in input_dict:
        input_dict[payment_col] = 1

    # 4. Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # 5. Predict churn probability
    churn_proba = model.predict_proba(input_df)[0][1]
    churn_percentage = churn_proba * 100
    
    st.markdown("---")
    st.subheader(f"Calculated Churn Probability: **{churn_percentage:.1f}%**")
    
    if churn_proba > 0.5:
        st.error("⚠️ **High Churn Risk!** Customer is likely to leave. Consider offering a retention discount or contract upgrade.")
    else:
        st.success("✅ **Low Churn Risk.** Customer behavior aligns with long-term retention.")