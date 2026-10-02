import sys
import os

# Enable project-root imports
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from src.recommendations import ChurnRetentionEngine


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction & Retention Intelligence",
    page_icon="🔮",
    layout="wide"
)


# ============================================================
# LOAD MODEL ARTIFACT
# ============================================================

@st.cache_resource
def load_artifact():

    artifact_path = os.path.join(
        "models",
        "pipeline.pkl"
    )

    if not os.path.exists(artifact_path):
        return None, None, None

    artifact = joblib.load(
        artifact_path
    )

    # Canonical artifact created by train.py
    if not isinstance(artifact, dict):
        raise ValueError(
            "Invalid model artifact. "
            "Please run python src/train.py again."
        )

    if "pipeline" not in artifact:
        raise ValueError(
            "Model artifact does not contain "
            "the trained pipeline."
        )

    if "optimal_threshold" not in artifact:
        raise ValueError(
            "Model artifact does not contain "
            "the optimized threshold."
        )

    pipeline = artifact["pipeline"]
    threshold = float(
        artifact["optimal_threshold"]
    )

    engine = ChurnRetentionEngine(
        artifact_path=artifact_path
    )

    return pipeline, threshold, engine


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.title(
        "🔮 Customer Churn Prediction & "
        "Retention Intelligence"
    )

    st.markdown(
        """
        End-to-end machine learning system for
        customer churn prediction, model explainability,
        and rule-based retention recommendations.
        """
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        pipeline, threshold, engine = load_artifact()

    except Exception as e:

        st.error(
            f"⚠️ Unable to load model artifact: {e}"
        )

        st.info(
            "Run `python src/train.py` to recreate "
            "the canonical model artifact."
        )

        return

    if pipeline is None:

        st.error(
            "⚠️ Model artifact not found at "
            "`models/pipeline.pkl`."
        )

        st.info(
            "Run `python src/train.py` first."
        )

        return


    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.title("Navigation")

    app_mode = st.sidebar.radio(
        "Select Interface Mode",
        [
            "Single Customer Assessment",
            "Batch Customer Analysis"
        ]
    )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        f"Locked Decision Threshold: "
        f"{threshold:.4f}"
    )

    st.sidebar.caption(
        "Threshold optimized using out-of-fold "
        "predictions and F2-score."
    )


    # ========================================================
    # SINGLE CUSTOMER ASSESSMENT
    # ========================================================

    if app_mode == "Single Customer Assessment":

        st.header(
            "👤 Single Customer Risk Assessment"
        )

        st.caption(
            "Enter customer attributes to evaluate "
            "real-time churn risk."
        )


        # ----------------------------------------------------
        # INPUT SECTION
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # ACCOUNT INFORMATION
        # ----------------------------------------------------

        with col1:

            st.subheader("Account Information")

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=120,
                value=6
            )

            contract = st.selectbox(
                "Contract Type",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                [
                    "Yes",
                    "No"
                ]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=300.0,
                value=85.50
            )

            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=513.00
            )


        # ----------------------------------------------------
        # SERVICES
        # ----------------------------------------------------

        with col2:

            st.subheader("Services Subscribed")

            phone_service = st.selectbox(
                "Phone Service",
                [
                    "Yes",
                    "No"
                ]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "No",
                    "Yes",
                    "No phone service"
                ]
            )

            internet_service = st.selectbox(
                "Internet Service",
                [
                    "Fiber optic",
                    "DSL",
                    "No"
                ]
            )

            online_security = st.selectbox(
                "Online Security",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            online_backup = st.selectbox(
                "Online Backup",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )


        # ----------------------------------------------------
        # ADDITIONAL SERVICES + DEMOGRAPHICS
        # ----------------------------------------------------

        with col3:

            st.subheader(
                "Additional Services & Demographics"
            )

            device_protection = st.selectbox(
                "Device Protection",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            tech_support = st.selectbox(
                "Tech Support",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            gender = st.selectbox(
                "Gender",
                [
                    "Female",
                    "Male"
                ]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [
                    0,
                    1
                ]
            )

            partner = st.selectbox(
                "Partner",
                [
                    "No",
                    "Yes"
                ]
            )

            dependents = st.selectbox(
                "Dependents",
                [
                    "No",
                    "Yes"
                ]
            )


        # ====================================================
        # PREDICTION
        # ====================================================

        if st.button(
            "Evaluate Churn Risk",
            type="primary"
        ):

            input_df = pd.DataFrame(
                [{
                    "gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges
                }]
            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            churn_proba = float(
                pipeline.predict_proba(
                    input_df
                )[0][1]
            )

            prediction = int(
                churn_proba >= threshold
            )


            # ------------------------------------------------
            # RISK CLASSIFICATION
            # ------------------------------------------------

            risk_tier = engine.categorize_risk(
                churn_proba
            )

            recommendation = (
                engine.generate_recommendation(
                    input_df.iloc[0].to_dict(),
                    churn_proba,
                    risk_tier
                )
            )


            # =================================================
            # RESULTS
            # =================================================

            st.markdown("---")

            st.subheader(
                "📊 Assessment Results"
            )

            res_col1, res_col2, res_col3, res_col4 = (
                st.columns(4)
            )


            with res_col1:

                st.metric(
                    "Predicted Churn Probability",
                    f"{churn_proba:.1%}"
                )


            with res_col2:

                st.metric(
                    "Classification",
                    "Churn"
                    if prediction == 1
                    else "No Churn"
                )


            with res_col3:

                st.metric(
                    "Risk Tier",
                    risk_tier
                )


            with res_col4:

                st.metric(
                    "Decision Threshold",
                    f"{threshold:.4f}"
                )


            # ------------------------------------------------
            # RISK MESSAGE
            # ------------------------------------------------

            if risk_tier in [
                "High Risk",
                "Critical Risk"
            ]:

                st.error(
                    f"🚨 **High Attention Needed:** "
                    f"Customer is classified as "
                    f"**{risk_tier}**."
                )

            elif risk_tier == "Medium Risk":

                st.warning(
                    f"⚠️ **Moderate Risk:** "
                    f"Customer falls into the "
                    f"**{risk_tier}** category."
                )

            else:

                st.success(
                    "✅ **Low Risk:** "
                    "Customer is currently stable."
                )


            # =================================================
            # SHAP LOCAL EXPLANATION
            # =================================================

            st.markdown("---")

            st.subheader(
                "💡 Why did the model make this prediction?"
            )

            st.caption(
                "SHAP explains which features contributed "
                "most to this individual prediction. "
                "These are model explanations, not causal claims."
            )


            try:

                classifier = pipeline.named_steps[
                    "classifier"
                ]

                preprocessor = pipeline.named_steps[
                    "preprocessor"
                ]


                # Transform customer input
                transformed_input = (
                    preprocessor.transform(
                        input_df
                    )
                )


                # Feature names after preprocessing
                feature_names = (
                    preprocessor
                    .get_feature_names_out()
                )


                # SHAP explainer
                explainer = shap.TreeExplainer(
                    classifier
                )

                shap_output = explainer(
                    transformed_input
                )


                # Extract values
                if hasattr(
                    shap_output,
                    "values"
                ):

                    shap_values = (
                        shap_output.values
                    )

                else:

                    shap_values = shap_output


                # Handle single prediction shape
                if shap_values.ndim == 2:

                    shap_values = (
                        shap_values[0]
                    )


                # Top contributing features
                top_indices = (
                    np.argsort(
                        np.abs(shap_values)
                    )[-10:]
                )


                display_names = [
                    feature_names[i]
                    .replace(
                        "cat__",
                        ""
                    )
                    .replace(
                        "num__",
                        ""
                    )
                    for i in top_indices
                ]


                top_values = (
                    shap_values[top_indices]
                )


                # Create plot
                fig, ax = plt.subplots(
                    figsize=(9, 5)
                )

                ax.barh(
                    display_names,
                    top_values
                )

                ax.axvline(
                    0,
                    linewidth=1
                )

                ax.set_xlabel(
                    "SHAP Value"
                )

                ax.set_title(
                    "Top Local Feature Contributions"
                )

                plt.tight_layout()

                st.pyplot(
                    fig,
                    clear_figure=True
                )

                plt.close(fig)


            except Exception as e:

                st.warning(
                    "Unable to generate SHAP "
                    f"explanation: {e}"
                )


            # =================================================
            # RETENTION RECOMMENDATIONS
            # =================================================

            st.markdown("---")

            st.subheader(
                "🛠 Retention Recommendations"
            )

            st.caption(
                "Recommendations are generated using "
                "a rule-based retention engine."
            )


            if isinstance(
                recommendation,
                dict
            ):

                st.markdown(
                    "##### Identified Drivers"
                )

                for driver in recommendation[
                    "Primary Churn Drivers"
                ]:

                    st.write(
                        f"- ⚠️ {driver}"
                    )


                st.markdown(
                    "##### Recommended Action Plan"
                )

                for action in recommendation[
                    "Recommended Action Plan"
                ]:

                    st.write(
                        f"- ✅ {action}"
                    )

            else:

                st.info(
                    recommendation
                )


    # ========================================================
    # BATCH CUSTOMER ANALYSIS
    # ========================================================

    elif app_mode == "Batch Customer Analysis":

        st.header(
            "📂 Batch Customer Analysis"
        )

        st.caption(
            "Upload a CSV dataset containing "
            "customer records."
        )


        uploaded_file = st.file_uploader(
            "Upload Customer Dataset (CSV)",
            type=["csv"]
        )


        if uploaded_file is not None:

            batch_df = pd.read_csv(
                uploaded_file
            )

            st.write(
                f"Uploaded {len(batch_df)} records."
            )


            if st.button(
                "Run Batch Risk Analysis",
                type="primary"
            ):

                with st.spinner(
                    "Processing batch predictions..."
                ):

                    results = (
                        engine.predict_customer(
                            batch_df
                        )
                    )


                    st.markdown("---")

                    st.subheader(
                        "📈 Risk Distribution"
                    )

                    st.dataframe(
                        results[
                            "Risk Tier"
                        ].value_counts()
                    )


                    st.subheader(
                        "📋 Customer Retention Roster"
                    )

                    merged_df = pd.concat(
                        [
                            batch_df,
                            results
                        ],
                        axis=1
                    )

                    st.dataframe(
                        merged_df,
                        use_container_width=True
                    )


                    # ------------------------------------------------
                    # Download report
                    # ------------------------------------------------

                    csv_data = (
                        merged_df
                        .to_csv(
                            index=False
                        )
                        .encode("utf-8")
                    )


                    st.download_button(
                        label=(
                            "⬇️ Download "
                            "Retention Report (CSV)"
                        ),
                        data=csv_data,
                        file_name=(
                            "retention_action_report.csv"
                        ),
                        mime="text/csv"
                    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()