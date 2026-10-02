import sys
import os

# Enable path resolution for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np


class ChurnRetentionEngine:

    def __init__(self, artifact_path="models/pipeline.pkl"):
        artifact = joblib.load(artifact_path)

        if not isinstance(artifact, dict):
            raise ValueError(
                "Invalid model artifact. Expected a dictionary containing "
                "'pipeline' and 'optimal_threshold'."
            )

        self.pipeline = artifact["pipeline"]
        self.optimal_threshold = artifact["optimal_threshold"]

    def categorize_risk(self, proba):

        if proba < self.optimal_threshold:
            return "Low Risk"

        elif proba < 0.40:
            return "Medium Risk"

        elif proba < 0.70:
            return "High Risk"

        else:
            return "Critical Risk"

    def generate_recommendation(self, customer_row, proba, risk_tier):

        if risk_tier == "Low Risk":
            return {
                "Primary Churn Drivers": [],
                "Recommended Action Plan": [
                    "No aggressive intervention required. Include in standard nurture communications."
                ]
            }

        drivers = []
        actions = []

        contract = customer_row.get("Contract", "")

        if contract == "Month-to-month":
            drivers.append(
                "Month-to-month contract structure"
            )

            actions.append(
                "Offer 15% discount on 1-Year or 2-Year Contract upgrade."
            )

        internet = customer_row.get("InternetService", "")
        tech_support = customer_row.get("TechSupport", "")
        security = customer_row.get("OnlineSecurity", "")

        if internet == "Fiber optic":

            if tech_support == "No":
                drivers.append(
                    "Fiber Optic service lacking Technical Support"
                )

                actions.append(
                    "Provide 3 months free Premium Tech Support."
                )

            if security == "No":
                drivers.append(
                    "Fiber Optic service lacking Security Suite"
                )

                actions.append(
                    "Bundle complimentary Security Suite."
                )

        payment = customer_row.get("PaymentMethod", "")

        if payment == "Electronic check":
            drivers.append(
                "Manual Electronic Check payment friction"
            )

            actions.append(
                "Offer $5 monthly bill credit for switching to Auto-Pay (ACH/Credit Card)."
            )

        tenure = customer_row.get("tenure", 0)
        monthly = customer_row.get("MonthlyCharges", 0)

        try:
            tenure = float(tenure)
        except (ValueError, TypeError):
            tenure = 0

        try:
            monthly = float(monthly)
        except (ValueError, TypeError):
            monthly = 0

        if tenure <= 12 and monthly > 70:
            drivers.append(
                "High early tenure pricing pressure"
            )

            actions.append(
                "Apply a $10/month loyalty credit for the next 6 months."
            )

        if not drivers:
            drivers.append(
                "General usage pattern instability"
            )

            actions.append(
                "Schedule a proactive customer success call to review account satisfaction."
            )

        return {
            "Primary Churn Drivers": drivers,
            "Recommended Action Plan": actions
        }

    def _prepare_customer_data(self, customer_df):

        customer_df = customer_df.copy()

        # Clean column names
        customer_df.columns = customer_df.columns.str.strip()

        # Remove columns that were not used during training
        customer_df = customer_df.drop(
            columns=["customerID", "Churn"],
            errors="ignore"
        )

        # Clean numeric columns
        numeric_columns = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        for col in numeric_columns:

            if col in customer_df.columns:

                customer_df[col] = pd.to_numeric(
                    customer_df[col].astype(str).str.strip(),
                    errors="coerce"
                )

        return customer_df

    def predict_customer(self, customer_df):

        # Keep original data for recommendations
        original_df = customer_df.copy()

        # Prepare data for ML pipeline
        prepared_df = self._prepare_customer_data(
            customer_df
        )

        # Predict churn probabilities
        probs = self.pipeline.predict_proba(
            prepared_df
        )[:, 1]

        results = []

        for idx, p in enumerate(probs):

            row = original_df.iloc[idx].to_dict()

            tier = self.categorize_risk(p)

            rec = self.generate_recommendation(
                row,
                p,
                tier
            )

            results.append({
                "Churn Probability": round(
                    float(p),
                    4
                ),
                "Risk Tier": tier,
                "Threshold Applied": self.optimal_threshold,
                "Recommendation": rec
            })

        return pd.DataFrame(results)


if __name__ == "__main__":

    if os.path.exists("models/pipeline.pkl"):

        engine = ChurnRetentionEngine(
            "models/pipeline.pkl"
        )

        print(
            f"Engine initialized with optimal threshold = "
            f"{engine.optimal_threshold}"
        )

    else:

        print(
            "models/pipeline.pkl not found. "
            "Run src/train.py first."
        )