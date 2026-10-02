import sys
import os

# Enable path resolution for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, fbeta_score, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

from src.preprocessing import create_preprocessing_pipeline

def find_optimal_threshold(y_true, y_probs, beta=2.0):
    thresholds = np.linspace(0.01, 0.99, 1000)
    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score

def main():
    print("🚀 Starting Churn Prediction Training Pipeline...")

    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(data_path):
        data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'. Please place the Telco CSV file in the data/ folder.")

    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = create_preprocessing_pipeline(X_train)
    
    from sklearn.pipeline import Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])

    param_distributions = {
        'classifier__n_estimators': [100, 150, 200],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__reg_alpha': [0, 0.1, 1.0],
        'classifier__reg_lambda': [1.0, 2.0, 5.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1
    )

    print("🔍 Executing Hyperparameter Search on Training Set...")
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    print(f"✅ Best Cross-Validation ROC-AUC: {search.best_score_:.4f}")

    print("⚡ Optimizing Decision Threshold via Out-of-Fold Predictions...")
    oof_probs = np.zeros(len(X_train))
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_X_tr, fold_y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        fold_X_val = X_train.iloc[val_idx]
        
        fold_pipeline = Pipeline([
            ('preprocessor', create_preprocessing_pipeline(fold_X_tr)),
            ('classifier', XGBClassifier(**search.best_params_, random_state=42, eval_metric='logloss'))
        ])
        fold_pipeline.fit(fold_X_tr, fold_y_tr)
        oof_probs[val_idx] = fold_pipeline.predict_proba(fold_X_val)[:, 1]

    optimal_threshold, oof_f2 = find_optimal_threshold(y_train, oof_probs, beta=2.0)
    print(f"🎯 Optimal Threshold: {optimal_threshold:.4f} | Training OOF F2 Score: {oof_f2:.4f}")

    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    test_preds_opt = (test_probs >= optimal_threshold).astype(int)

    print("\n--- 📊 Untouched Test Set Evaluation ---")
    print(f"ROC-AUC    : {roc_auc_score(y_test, test_probs):.4f}")
    print(f"Accuracy   : {accuracy_score(y_test, test_preds_opt):.4f}")
    print(f"Precision  : {precision_score(y_test, test_preds_opt):.4f}")
    print(f"Recall     : {recall_score(y_test, test_preds_opt):.4f}")
    print(f"F1-score   : {f1_score(y_test, test_preds_opt):.4f}")
    print(f"F2-score   : {fbeta_score(y_test, test_preds_opt, beta=2):.4f}")

    os.makedirs("models", exist_ok=True)
    artifact = {
        "pipeline": best_pipeline,
        "optimal_threshold": float(optimal_threshold),
        "best_params": search.best_params_
    }
    artifact_path = os.path.join("models", "pipeline.pkl")
    joblib.dump(artifact, artifact_path)
    print(f"\n💾 Successfully saved canonical artifact to '{artifact_path}'!")

if __name__ == "__main__":
    main()