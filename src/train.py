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


import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.preprocessing import create_preprocessing_pipeline


# =========================================================
# FIND OPTIMAL THRESHOLD
# =========================================================

def find_optimal_threshold(
    y_true,
    y_probs,
    beta=2.0
):
    """
    Find the probability threshold that maximizes
    the F-beta score.

    Beta = 2 gives more importance to recall.
    """

    thresholds = np.linspace(
        0.01,
        0.99,
        1000
    )

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:

        predictions = (
            y_probs >= threshold
        ).astype(int)

        score = fbeta_score(
            y_true,
            predictions,
            beta=beta,
            zero_division=0
        )

        if score > best_score:

            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "🚀 Starting Churn Prediction Training Pipeline..."
    )


    # =====================================================
    # 1. LOAD DATA
    # =====================================================

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
            f"Dataset not found at '{data_path}'. "
            "Please place the Telco CSV file in the data/ folder."
        )


    df = pd.read_csv(data_path)


    # =====================================================
    # 2. DATA CLEANING
    # =====================================================

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )


    if "customerID" in df.columns:

        df = df.drop(
            columns=["customerID"]
        )


    # =====================================================
    # 3. FEATURES + TARGET
    # =====================================================

    X = df.drop(
        columns=["Churn"]
    )

    y = df["Churn"].map(
        {
            "Yes": 1,
            "No": 0
        }
    )


    # =====================================================
    # 4. TRAIN / TEST SPLIT
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
    # 5. PREPROCESSING
    # =====================================================

    preprocessor = create_preprocessing_pipeline(
        X_train
    )


    # =====================================================
    # 6. COMPLETE PIPELINE
    # =====================================================

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",

                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1
                )
            )
        ]
    )


    # =====================================================
    # 7. HYPERPARAMETER SEARCH
    # =====================================================

    param_distributions = {

        "classifier__n_estimators": [
            100,
            150,
            200
        ],

        "classifier__max_depth": [
            3,
            4,
            5,
            6
        ],

        "classifier__learning_rate": [
            0.01,
            0.05,
            0.1
        ],

        "classifier__subsample": [
            0.7,
            0.8,
            1.0
        ],

        "classifier__colsample_bytree": [
            0.7,
            0.8,
            1.0
        ],

        "classifier__gamma": [
            0,
            0.1,
            0.2
        ],

        "classifier__reg_alpha": [
            0,
            0.1,
            1.0
        ],

        "classifier__reg_lambda": [
            1.0,
            2.0,
            5.0
        ]
    }


    # =====================================================
    # 8. STRATIFIED 5-FOLD CV
    # =====================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    # =====================================================
    # 9. RANDOMIZED SEARCH
    # =====================================================

    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=param_distributions,

        n_iter=15,

        scoring="roc_auc",

        cv=cv,

        random_state=42,

        n_jobs=-1,

        verbose=1
    )


    print(
        "\n🔍 Executing Hyperparameter Search..."
    )


    search.fit(
        X_train,
        y_train
    )


    best_pipeline = search.best_estimator_


    print(
        f"\n✅ Best Cross-Validation ROC-AUC: "
        f"{search.best_score_:.4f}"
    )


    print(
        "\nBest Hyperparameters:"
    )


    for parameter, value in search.best_params_.items():

        print(
            f"  {parameter}: {value}"
        )


    # =====================================================
    # 10. OUT-OF-FOLD PREDICTIONS
    #
    # IMPORTANT:
    #
    # We use the COMPLETE tuned pipeline here.
    #
    # This prevents the classifier__ parameter bug.
    # =====================================================

    print(
        "\n⚡ Generating Out-of-Fold "
        "Predictions for Threshold Optimization..."
    )


    oof_probs = cross_val_predict(

        best_pipeline,

        X_train,

        y_train,

        cv=cv,

        method="predict_proba",

        n_jobs=-1

    )[:, 1]


    # =====================================================
    # 11. OPTIMIZE THRESHOLD
    # =====================================================

    optimal_threshold, oof_f2 = (
        find_optimal_threshold(

            y_train,

            oof_probs,

            beta=2.0
        )
    )


    print(
        f"\n🎯 Optimal Threshold: "
        f"{optimal_threshold:.4f}"
    )

    print(
        f"Training OOF F2 Score: "
        f"{oof_f2:.4f}"
    )


    # =====================================================
    # 12. TEST PROBABILITIES
    #
    # Test set is untouched until this point.
    # =====================================================

    test_probs = best_pipeline.predict_proba(
        X_test
    )[:, 1]


    # =====================================================
    # 13. DEFAULT THRESHOLD = 0.5
    # =====================================================

    default_preds = (
        test_probs >= 0.5
    ).astype(int)


    default_precision = precision_score(
        y_test,
        default_preds,
        zero_division=0
    )

    default_recall = recall_score(
        y_test,
        default_preds,
        zero_division=0
    )

    default_f1 = f1_score(
        y_test,
        default_preds,
        zero_division=0
    )

    default_f2 = fbeta_score(
        y_test,
        default_preds,
        beta=2.0,
        zero_division=0
    )


    # =====================================================
    # 14. OPTIMIZED THRESHOLD
    # =====================================================

    optimized_preds = (
        test_probs >= optimal_threshold
    ).astype(int)


    optimized_precision = precision_score(
        y_test,
        optimized_preds,
        zero_division=0
    )

    optimized_recall = recall_score(
        y_test,
        optimized_preds,
        zero_division=0
    )

    optimized_f1 = f1_score(
        y_test,
        optimized_preds,
        zero_division=0
    )

    optimized_f2 = fbeta_score(
        y_test,
        optimized_preds,
        beta=2.0,
        zero_division=0
    )


    # =====================================================
    # 15. FINAL TEST METRICS
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        optimized_preds
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probs
    )

    pr_auc = average_precision_score(
        y_test,
        test_probs
    )


    # =====================================================
    # 16. CONFUSION MATRIX
    # =====================================================

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        optimized_preds
    ).ravel()


    # =====================================================
    # 17. FINAL RESULTS
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "📊 FINAL TEST SET RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Locked Threshold : {optimal_threshold:.4f}"
    )

    print(
        f"Accuracy         : {accuracy:.4f}"
    )

    print(
        f"Precision        : {optimized_precision:.4f}"
    )

    print(
        f"Recall           : {optimized_recall:.4f}"
    )

    print(
        f"F1-score         : {optimized_f1:.4f}"
    )

    print(
        f"F2-score         : {optimized_f2:.4f}"
    )

    print(
        f"ROC-AUC          : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC           : {pr_auc:.4f}"
    )

    print(
        "=" * 70
    )


    # =====================================================
    # 18. CONFUSION MATRIX
    # =====================================================

    print(
        "\n🔢 CONFUSION MATRIX"
    )

    print(
        "-" * 40
    )

    print(
        f"True Negatives  (TN): {tn}"
    )

    print(
        f"False Positives (FP): {fp}"
    )

    print(
        f"False Negatives (FN): {fn}"
    )

    print(
        f"True Positives  (TP): {tp}"
    )

    print(
        "-" * 40
    )


    # =====================================================
    # 19. THRESHOLD COMPARISON
    # =====================================================

    print(
        "\n⚖️ THRESHOLD COMPARISON"
    )

    print(
        "-" * 70
    )

    print(
        f"{'Metric':<15}"
        f"{'Default 0.500':<20}"
        f"{'Optimized':<20}"
    )

    print(
        "-" * 70
    )

    print(
        f"{'Precision':<15}"
        f"{default_precision:<20.4f}"
        f"{optimized_precision:<20.4f}"
    )

    print(
        f"{'Recall':<15}"
        f"{default_recall:<20.4f}"
        f"{optimized_recall:<20.4f}"
    )

    print(
        f"{'F1-score':<15}"
        f"{default_f1:<20.4f}"
        f"{optimized_f1:<20.4f}"
    )

    print(
        f"{'F2-score':<15}"
        f"{default_f2:<20.4f}"
        f"{optimized_f2:<20.4f}"
    )

    print(
        "-" * 70
    )


    # =====================================================
    # 20. SAVE MODEL
    # =====================================================

    os.makedirs(
        "models",
        exist_ok=True
    )


    artifact = {

        "pipeline": best_pipeline,

        "optimal_threshold": float(
            optimal_threshold
        ),

        "best_params": search.best_params_

    }


    artifact_path = os.path.join(
        "models",
        "pipeline.pkl"
    )


    joblib.dump(
        artifact,
        artifact_path
    )


    print(
        f"\n💾 Successfully saved canonical "
        f"artifact to '{artifact_path}'!"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()