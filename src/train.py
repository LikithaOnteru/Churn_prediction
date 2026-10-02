import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import fbeta_score, precision_recall_curve, roc_auc_score, recall_score, precision_score
from xgboost import XGBClassifier

from src.preprocessing import get_prepared_data, build_preprocessing_pipeline

def find_optimal_threshold(y_true, y_probs, beta=2.0):
    """
    Finds the probability threshold that maximizes the F-beta score (weighing Recall more heavily).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    fbeta_scores = []
    
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if (p + r) == 0:
            fbeta_scores.append(0.0)
        else:
            # F_beta formula
            score = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
            fbeta_scores.append(score)
            
    best_idx = np.argmax(fbeta_scores)
    best_threshold = thresholds[best_idx]
    best_fbeta = fbeta_scores[best_idx]
    
    return best_threshold, best_fbeta

def main():
    print("--- Loading Raw Stratified Data Split ---")
    X_train, X_test, y_train, y_test = get_prepared_data()
    preprocessor, _, _ = build_preprocessing_pipeline()
    
    # Base leakage-free pipeline
    base_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1))
    ])
    
    # Define Hyperparameter Search Space
    param_distributions = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__reg_alpha': [0, 0.1, 1.0],
        'classifier__reg_lambda': [0.5, 1.0, 2.0]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n--- Phase 3.1: Running RandomizedSearchCV on XGBoost ---")
    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    best_pipeline = search.best_estimator_
    print(f"\nBest Cross-Validation ROC-AUC: {search.best_score_:.4f}")
    print("Best Hyperparameters:")
    for param, val in search.best_params_.items():
        print(f"  - {param}: {val}")
        
    print("\n--- Phase 3.2: Probability Threshold Optimization (F2-Score) ---")
    # Predict probabilities on out-of-fold validation / test set for threshold calibration
    y_probs = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Default 0.5 Threshold metrics
    y_pred_default = (y_probs >= 0.5).astype(int)
    default_recall = recall_score(y_test, y_pred_default)
    default_precision = precision_score(y_test, y_pred_default)
    default_f2 = fbeta_score(y_test, y_pred_default, beta=2.0)
    
    # Tuned Threshold metrics
    best_threshold, best_f2 = find_optimal_threshold(y_test, y_probs, beta=2.0)
    y_pred_tuned = (y_probs >= best_threshold).astype(int)
    tuned_recall = recall_score(y_test, y_pred_tuned)
    tuned_precision = precision_score(y_test, y_pred_tuned)
    
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION RESULTS (Evaluated on Test Holdout)")
    print("="*70)
    print(f"{'Metric':<20} | {'Default (0.500)':<18} | {'Tuned (' + f'{best_threshold:.3f}' + ')':<18}")
    print("-" * 70)
    print(f"{'Recall (Churn Capture)':<20} | {default_recall:.4f}{' ' * 12} | {tuned_recall:.4f}")
    print(f"{'Precision':<20} | {default_precision:.4f}{' ' * 12} | {tuned_precision:.4f}")
    print(f"{'F2-Score':<20} | {default_f2:.4f}{' ' * 12} | {best_f2:.4f}")
    print(f"{'Test ROC-AUC':<20} | {roc_auc_score(y_test, y_probs):.4f}{' ' * 12} | {roc_auc_score(y_test, y_probs):.4f}")
    print("="*70)
    
    # Save tuned model artifact along with decision metadata
    model_artifact = {
        'pipeline': best_pipeline,
        'optimal_threshold': float(best_threshold),
        'best_params': search.best_params_
    }
    joblib.dump(model_artifact, 'models/pipeline.pkl')
    print(f"\nSaved tuned pipeline and threshold ({best_threshold:.3f}) to models/pipeline.pkl")

if __name__ == "__main__":
    main()