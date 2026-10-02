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
        if isinstance(artifact, dict):
            self.pipeline = artifact["pipeline"]
            self.optimal_threshold = artifact.get("optimal_threshold", 0.1836)
        else:
            self.pipeline = artifact
            self.optimal_threshold = 0.1836

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
            return "No aggressive intervention required. Include in standard nurture communications."

        drivers = []
        actions = []

        contract = customer_row.get('Contract', '')
        if contract == 'Month-to-month':
            drivers.append("Month-to-month contract structure")
            actions.append("Offer 15% discount on 1-Year or 2-Year Contract upgrade.")

        internet = customer_row.get('InternetService', '')
        tech_support = customer_row.get('TechSupport', '')
        security = customer_row.get('OnlineSecurity', '')
        
        if internet == 'Fiber optic':
            if tech_support == 'No':
                drivers.append("Fiber Optic service lacking Technical Support")
                actions.append("Provide 3 months free Premium Tech Support.")
            if security == 'No':
                drivers.append("Fiber Optic service lacking Security Suite")
                actions.append("Bundle complimentary Security Suite.")

        payment = customer_row.get('PaymentMethod', '')
        if payment == 'Electronic check':
            drivers.append("Manual Electronic Check payment friction")
            actions.append("Offer $5 monthly bill credit for switching to Auto-Pay (ACH/Credit Card).")

        tenure = customer_row.get('tenure', 0)
        monthly = customer_row.get('MonthlyCharges', 0)
        if tenure <= 12 and monthly > 70:
            drivers.append("High early tenure pricing pressure")
            actions.append("Apply a $10/month loyalty credit for the next 6 months.")

        if not drivers:
            drivers.append("General usage pattern instability")
            actions.append("Schedule a proactive customer success call to review account satisfaction.")

        return {
            'Primary Churn Drivers': drivers,
            'Recommended Action Plan': actions
        }

    def predict_customer(self, customer_df):
        probs = self.pipeline.predict_proba(customer_df)[:, 1]
        results = []
        for idx, p in enumerate(probs):
            row = customer_df.iloc[idx].to_dict()
            tier = self.categorize_risk(p)
            rec = self.generate_recommendation(row, p, tier)
            results.append({
                'Churn Probability': round(float(p), 4),
                'Risk Tier': tier,
                'Threshold Applied': self.optimal_threshold,
                'Recommendation': rec
            })
        return pd.DataFrame(results)

if __name__ == "__main__":
    if os.path.exists("models/pipeline.pkl"):
        engine = ChurnRetentionEngine("models/pipeline.pkl")
        print(f"Engine initialized with optimal threshold = {engine.optimal_threshold}")
    else:
        print("models/pipeline.pkl not found. Run src/train.py first.")