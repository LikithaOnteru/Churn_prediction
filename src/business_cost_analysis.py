import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/pipeline.pkl"


# Assumed business costs
INTERVENTION_COST = 10
MISSED_CHURN_COST = 100


def load_data():

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    return X, y


def main():

    print("🚀 Starting Business Cost Analysis...")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    artifact = joblib.load(MODEL_PATH)

    pipeline = artifact["pipeline"]
    locked_threshold = artifact["optimal_threshold"]

    print(
        f"Locked F2 Threshold: "
        f"{locked_threshold:.4f}"
    )

    print(
        f"Assumed intervention cost: "
        f"${INTERVENTION_COST}"
    )

    print(
        f"Assumed missed-churn cost: "
        f"${MISSED_CHURN_COST}"
    )

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    X, y = load_data()

    # IMPORTANT:
    # Use the exact same untouched test split
    # used during final model evaluation.

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # Generate test probabilities
    # --------------------------------------------------

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------

    thresholds = np.arange(
        0.05,
        0.51,
        0.01
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions
        ).ravel()

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        f2 = fbeta_score(
            y_test,
            predictions,
            beta=2,
            zero_division=0
        )

        intervention_cost = (
            (tp + fp) * INTERVENTION_COST
        )

        missed_churn_cost = (
            fn * MISSED_CHURN_COST
        )

        total_cost = (
            intervention_cost +
            missed_churn_cost
        )

        results.append({
            "Threshold": round(threshold, 2),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "F2": f2,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Intervention_Cost": intervention_cost,
            "Missed_Churn_Cost": missed_churn_cost,
            "Total_Cost": total_cost
        })

    results_df = pd.DataFrame(results)

    # --------------------------------------------------
    # Find cost-minimizing threshold
    # --------------------------------------------------

    cost_optimal = results_df.loc[
        results_df["Total_Cost"].idxmin()
    ]

    # --------------------------------------------------
    # Show important thresholds
    # --------------------------------------------------

    print("\n" + "=" * 95)
    print("💰 BUSINESS COST THRESHOLD ANALYSIS")
    print("=" * 95)

    selected_thresholds = results_df[
        results_df["Threshold"].isin(
            [
                0.10,
                0.15,
                0.17,
                0.20,
                0.25,
                0.30,
                0.40,
                0.50
            ]
        )
    ]

    print(
        selected_thresholds[
            [
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "F2",
                "FP",
                "FN",
                "Total_Cost"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n" + "-" * 95)

    print(
        f"🎯 F2-Optimized Threshold: "
        f"{locked_threshold:.4f}"
    )

    print(
        f"💰 Cost-Minimizing Threshold: "
        f"{cost_optimal['Threshold']:.2f}"
    )

    print(
        f"💵 Minimum Assumed Total Cost: "
        f"${cost_optimal['Total_Cost']:.2f}"
    )

    print("-" * 95)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = (
        "results/business_cost_analysis.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n💾 Results saved to: "
        f"{output_path}"
    )

    print(
        "\n⚠️ Business costs are illustrative assumptions, "
        "not real company costs."
    )

    print(
        "\n✅ Business cost analysis completed."
    )


if __name__ == "__main__":
    main()