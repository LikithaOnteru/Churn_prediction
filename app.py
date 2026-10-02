import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.recommendations import generate_retention_recommendations


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn & Retention Intelligence",
    layout="wide"
)


# =========================================================
# LOAD MODEL ARTIFACT
# =========================================================

@st.cache_resource
def load_artifacts():

    artifact = joblib.load(
        "models/pipeline.pkl"
    )

    pipeline = artifact["pipeline"]
    threshold = artifact["optimal_threshold"]

    return pipeline, threshold


try:

    pipeline, threshold = load_artifacts()

except Exception as e:

    st.error(
        "Error loading model artifact. "
        "Please run `python src/train.py` first."
    )

    st.stop()


# =========================================================
# PAGE TITLE
# =========================================================

st.title(
    "📊 Customer Churn Prediction & Retention Dashboard"
)

st.write(
    "Enter customer attributes to calculate churn probability, "
    "identify risk level, view feature contributions, and receive "
    "rule-based retention recommendations."
)


# =========================================================
# SIDEBAR — CUSTOMER INPUT
# =========================================================

st.sidebar.header(
    "Customer Profile Input"
)


tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)


monthly_charges = st.sidebar.number_input(
    "Monthly Charges ($)",
    18.0,
    120.0,
    65.0
)


total_charges = st.sidebar.number_input(
    "Total Charges ($)",
    0.0,
    9000.0,
    float(tenure * monthly_charges)
)


gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)


senior = st.sidebar.selectbox(
    "Senior Citizen",
    ["0", "1"]
)


partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)


dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)


contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)


internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)


payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)


phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)


multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)


security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)


backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)


device = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)


tech = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)


tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)


movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_dict = {

    "gender": gender,

    "SeniorCitizen": senior,

    "Partner": partner,

    "Dependents": dependents,

    "tenure": tenure,

    "PhoneService": phone,

    "MultipleLines": multiple_lines,

    "InternetService": internet,

    "OnlineSecurity": security,

    "OnlineBackup": backup,

    "DeviceProtection": device,

    "TechSupport": tech,

    "StreamingTV": tv,

    "StreamingMovies": movies,

    "Contract": contract,

    "PaperlessBilling": paperless,

    "PaymentMethod": payment,

    "MonthlyCharges": monthly_charges,

    "TotalCharges": total_charges
}


input_df = pd.DataFrame(
    [input_dict]
)


# =========================================================
# CHURN PREDICTION
# =========================================================

if st.button(
    "Calculate Churn Risk",
    type="primary"
):

    # -----------------------------------------------------
    # Predict directly using the complete pipeline
    # -----------------------------------------------------

    churn_proba = pipeline.predict_proba(
        input_df
    )[0][1]


    # -----------------------------------------------------
    # Apply the optimized threshold
    # -----------------------------------------------------

    prediction = int(
        churn_proba >= threshold
    )


    # =====================================================
    # RISK SEGMENTATION
    # =====================================================

    if churn_proba < 0.30:

        risk_tier = "Low Risk"
        color = "green"

    elif churn_proba < 0.60:

        risk_tier = "Medium Risk"
        color = "orange"

    elif churn_proba < 0.80:

        risk_tier = "High Risk"
        color = "red"

    else:

        risk_tier = "Critical Risk"
        color = "darkred"


    # =====================================================
    # RESULTS
    # =====================================================

    col1, col2 = st.columns(
        [1, 1]
    )


    # -----------------------------------------------------
    # CHURN PROBABILITY
    # -----------------------------------------------------

    with col1:

        st.metric(
            label="Calculated Churn Probability",
            value=f"{churn_proba * 100:.1f}%"
        )


        st.markdown(
            f"""
            ### Risk Tier:
            <span style="color:{color}; font-size:24px;">
            {risk_tier}
            </span>
            """,
            unsafe_allow_html=True
        )


        if prediction == 1:

            st.warning(
                f"⚠️ Customer is flagged as likely to churn "
                f"at the optimized threshold of {threshold:.3f}."
            )

        else:

            st.success(
                f"✅ Customer is below the churn decision threshold "
                f"of {threshold:.3f}."
            )


    # =====================================================
    # RETENTION RECOMMENDATIONS
    # =====================================================

    with col2:

        st.subheader(
            "💡 Rule-Based Retention Recommendations"
        )


        recs = generate_retention_recommendations(
            input_dict,
            churn_proba
        )


        for recommendation in recs:

            st.write(
                recommendation
            )


    # =====================================================
    # SHAP EXPLANATION
    # =====================================================

    st.markdown("---")


    st.subheader(
        "🔍 Local SHAP Feature Contribution"
    )


    try:

        # Extract the trained XGBoost classifier
        model = pipeline.named_steps[
            "classifier"
        ]


        # Apply preprocessing only for SHAP
        preprocessor = pipeline.named_steps[
            "preprocessor"
        ]


        input_transformed = preprocessor.transform(
            input_df
        )


        # Get feature names
        try:

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

        except Exception:

            feature_names = None


        # Create SHAP explainer
        explainer = shap.TreeExplainer(
            model
        )


        shap_values = explainer(
            input_transformed
        )


        # Create readable SHAP explanation
        if feature_names is not None:

            shap_values.feature_names = (
                feature_names
            )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        shap.plots.bar(
            shap_values[0],
            max_display=7,
            show=False
        )


        st.pyplot(
            fig,
            clear_figure=True
        )


    except Exception as e:

        st.warning(
            "SHAP explanation could not be generated "
            "for this input."
        )