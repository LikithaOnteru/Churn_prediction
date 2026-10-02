import sys
import os

# =========================================================
# ENABLE PROJECT PATH
# =========================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
)

from xgboost import XGBClassifier

from src.preprocessing import (
    create_preprocessing_pipeline
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    data_path = os.path.join(
        "data",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    if not os.path.exists(data_path):

        data_path = (
            "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )

    if not os.path.exists(data_path):

        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'."
        )


    df = pd.read_csv(data_path)


    # -----------------------------------------------------
    # Clean TotalCharges
    # -----------------------------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = (
        df["TotalCharges"]
        .fillna(
            df["TotalCharges"].median()
        )
    )


    # -----------------------------------------------------
    # Remove customer ID
    # -----------------------------------------------------

    if "customerID" in df.columns:

        df = df.drop(
            columns=["customerID"]
        )


    # -----------------------------------------------------
    # Features and target
    # -----------------------------------------------------

    X = df.drop(
        columns=["Churn"]
    )

    y = df["Churn"].map(
        {
            "Yes": 1,
            "No": 0
        }
    )


    return X, y


# =========================================================
# BUILD MODEL PIPELINE
# =========================================================

def build_pipeline(
    X,
    model
):

    preprocessor = create_preprocessing_pipeline(
        X
    )


    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                model
            )
        ]
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "🚀 Starting Model Comparison..."
    )


    # =====================================================
    # 1. LOAD DATA
    # =====================================================

    X, y = load_data()


    # =====================================================
    # 2. TRAIN / TEST SPLIT
    #
    # Test set is NOT used for model selection.
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Test samples:     {len(X_test)}"
    )


    # =====================================================
    # 3. DEFINE MODELS
    # =====================================================

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.2,
                reg_alpha=0,
                reg_lambda=5.0,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1
            )
    }


    # =====================================================
    # 4. CROSS-VALIDATION
    # =====================================================

    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42
    )


    # =====================================================
    # 5. METRICS
    # =====================================================

    scoring = {

        "roc_auc":
            "roc_auc",

        "pr_auc":
            "average_precision",

        "precision":
            make_scorer(
                precision_score,
                zero_division=0
            ),

        "recall":
            make_scorer(
                recall_score,
                zero_division=0
            ),

        "f1":
            make_scorer(
                f1_score,
                zero_division=0
            ),

        "f2":
            make_scorer(
                fbeta_score,
                beta=2.0,
                zero_division=0
            )
    }


    # =====================================================
    # 6. RUN MODEL COMPARISON
    # =====================================================

    results = []


    for model_name, model in models.items():

        print(
            f"\n🔄 Evaluating {model_name}..."
        )


        pipeline = build_pipeline(
            X_train,
            model
        )


        scores = cross_validate(

            pipeline,

            X_train,

            y_train,

            cv=cv,

            scoring=scoring,

            n_jobs=-1,

            return_train_score=False
        )


        results.append(

            {

                "Model": model_name,

                "ROC-AUC":
                    scores[
                        "test_roc_auc"
                    ].mean(),

                "PR-AUC":
                    scores[
                        "test_pr_auc"
                    ].mean(),

                "Precision":
                    scores[
                        "test_precision"
                    ].mean(),

                "Recall":
                    scores[
                        "test_recall"
                    ].mean(),

                "F1":
                    scores[
                        "test_f1"
                    ].mean(),

                "F2":
                    scores[
                        "test_f2"
                    ].mean()
            }
        )


    # =====================================================
    # 7. CREATE RESULTS TABLE
    # =====================================================

    results_df = pd.DataFrame(
        results
    )


    # =====================================================
    # 8. DISPLAY RESULTS
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "📊 MODEL COMPARISON — 5-FOLD STRATIFIED CV"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(
        "=" * 80
    )


    # =====================================================
    # 9. SAVE RESULTS
    # =====================================================

    os.makedirs(
        "results",
        exist_ok=True
    )


    results_path = os.path.join(
        "results",
        "model_comparison.csv"
    )


    results_df.to_csv(
        results_path,
        index=False
    )


    print(
        f"\n💾 Results saved to:"
        f" {results_path}"
    )


    # =====================================================
    # 10. NOTE ABOUT TEST SET
    # =====================================================

    print(
        "\nℹ️ The test set was kept completely "
        "untouched during model comparison."
    )

    print(
        "The final XGBoost model will be evaluated "
        "on the test set separately."
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()