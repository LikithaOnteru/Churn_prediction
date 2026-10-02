import sys
import os

# Enable path resolution for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from src.recommendations import ChurnRetentionEngine

st.set_page_config(
    page_title="Customer Churn Prediction & Retention Intelligence",
    page_icon="🔮",
    layout="wide"
)

@st.cache_resource
def load_artifact():
    artifact_path = os.path.join("models", "pipeline.pkl")
    if not os.path.exists(artifact_path):
        return None, None, None
    
    artifact = joblib.load(artifact_path)
    if isinstance(artifact, dict):
        pipeline = artifact["pipeline"]
        threshold = artifact.get("optimal_threshold", 0.1836)
        engine = ChurnRetentionEngine(artifact_path=artifact_path)
        return pipeline, threshold, engine
    else:
        pipeline = artifact
        threshold = 0.1836
        engine = ChurnRetentionEngine(artifact_path=artifact_path)
        return pipeline, threshold, engine

def main():
    st.title("🔮 Customer Churn Prediction & Retention Intelligence")
    st.markdown("End-to-end Machine Learning inference, SHAP feature explainability, and prescriptive retention recommendations.")

    pipeline, threshold, engine = load_artifact()

    if pipeline is None:
        st.error("⚠️ Model artifact not found at `models/pipeline.pkl`. Please run `python src/train.py` first.")
        return

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Interface Mode", ["Single Customer Assessment", "Batch Customer Analysis"])

    if app_mode == "Single Customer Assessment":
        st.header("👤 Single Customer Risk Assessment")
        st.caption("Input customer attributes to evaluate real-time churn risk.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Account Info")
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=6)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=300.0, value=85.50)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=513.00)

        with col2:
            st.subheader("Services Subscribed")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

        with col3:
            st.subheader("Additional Services & Demographics")
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        if st.button("Evaluate Churn Risk", type="primary"):
            input_df = pd.DataFrame([{
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }])

            churn_proba = float(pipeline.predict_proba(input_df)[0][1])
            prediction = int(churn_proba >= threshold)
            risk_tier = engine.categorize_risk(churn_proba)
            recommendation = engine.generate_recommendation(input_df.iloc[0].to_dict(), churn_proba, risk_tier)

            st.markdown("---")
            st.subheader("📊 Assessment Results")

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric("Predicted Probability", f"{churn_proba:.1%}")
            with res_col2:
                st.metric("Classification Result", "Churn" if prediction == 1 else "No Churn")
            with res_col3:
                st.metric("Risk Classification Tier", risk_tier)
            with res_col4:
                st.metric("Decision Threshold", f"{threshold:.4f}")

            if risk_tier in ["High Risk", "Critical Risk"]:
                st.error(f"🚨 **High Attention Needed:** Customer is classified as **{risk_tier}**.")
            elif risk_tier == "Medium Risk":
                st.warning(f"⚠️ **Moderate Risk:** Customer falls into the **{risk_tier}** category.")
            else:
                st.success("✅ **Low Risk:** Customer is currently stable.")

            # SHAP Local Explanation
            st.markdown("---")
            st.subheader("💡 SHAP Feature Explainability")
            try:
                classifier = pipeline.named_steps["classifier"]
                preprocessor = pipeline.named_steps["preprocessor"]

                transformed_input = preprocessor.transform(input_df)
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except Exception:
                    feature_names = [f"Feature {i}" for i in range(transformed_input.shape[1])]

                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(transformed_input)

                fig, ax = plt.subplots(figsize=(8, 4))
                if isinstance(shap_values, list):
                    vals = shap_values[1][0]
                else:
                    vals = shap_values[0]

                top_indices = np.argsort(np.abs(vals))[-8:]
                ax.barh([feature_names[i].replace("cat__", "").replace("num__", "") for i in top_indices], vals[top_indices], color="#1f77b4")
                ax.set_xlabel("SHAP Value (Impact on Churn Probability)")
                ax.set_title("Top Local Feature Drivers")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Unable to generate SHAP explanation for this input profile: {str(e)}")

            # Prescriptive Recommendations
            st.markdown("---")
            st.subheader("🛠 Retention Recommendations")
            if isinstance(recommendation, dict):
                st.markdown("##### Identified Drivers")
                for d in recommendation['Primary Churn Drivers']:
                    st.write(f"- ⚠ {d}")
                st.markdown("##### Action Plan")
                for a in recommendation['Recommended Action Plan']:
                    st.write(f"- ✅ {a}")
            else:
                st.info(recommendation)

    elif app_mode == "Batch Customer Analysis":
        st.header("📂 Batch Customer Analysis")
        st.caption("Upload a CSV dataset containing customer records.")

        uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(batch_df)} records.")

            if st.button("Run Batch Risk Analysis"):
                with st.spinner("Processing batch predictions..."):
                    results = engine.predict_customer(batch_df)
                    st.markdown("---")
                    st.subheader("📈 Summary Overview")
                    st.dataframe(results['Risk Tier'].value_counts())

                    merged_df = pd.concat([batch_df, results], axis=1)
                    st.subheader("📋 Customer Retention Roster")
                    st.dataframe(merged_df)

                    csv_data = merged_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Retention Report (CSV)",
                        data=csv_data,
                        file_name="retention_action_report.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()