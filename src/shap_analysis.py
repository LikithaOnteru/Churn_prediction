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
import shap
import matplotlib.pyplot as plt


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/pipeline.pkl"


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

    return X


def main():

    print("🚀 Starting SHAP Analysis...")

    # --------------------------------------------------
    # Load model artifact
    # --------------------------------------------------

    artifact = joblib.load(MODEL_PATH)

    pipeline = artifact["pipeline"]
    threshold = artifact["optimal_threshold"]

    print(f"Locked Threshold: {threshold:.4f}")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    X = load_data()

    # Use a sample for SHAP to keep computation reasonable
    X_sample = X.sample(
        n=min(500, len(X)),
        random_state=42
    )

    # --------------------------------------------------
    # Extract preprocessing + classifier
    # --------------------------------------------------

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    # Transform data
    X_transformed = preprocessor.transform(X_sample)

    # Feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # --------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------

    print("\n🔍 Calculating SHAP values...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_transformed
    )

    # --------------------------------------------------
    # Global Feature Importance
    # --------------------------------------------------

    importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean_Absolute_SHAP": importance
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("📊 TOP 20 GLOBAL CHURN DRIVERS")
    print("=" * 70)

    print(
        importance_df.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------
    # Save feature importance
    # --------------------------------------------------

    os.makedirs("results", exist_ok=True)

    importance_path = "results/shap_feature_importance.csv"

    importance_df.to_csv(
        importance_path,
        index=False
    )

    print(
        f"\n💾 Feature importance saved to: "
        f"{importance_path}"
    )

    # --------------------------------------------------
    # SHAP Summary Plot
    # --------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    plot_path = "results/shap_summary.png"

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"📈 SHAP summary plot saved to: "
        f"{plot_path}"
    )

    print("\n✅ SHAP analysis completed successfully.")


if __name__ == "__main__":
    main()