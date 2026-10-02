import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, roc_auc_score

from preprocessing import load_and_clean_data, build_preprocessing_pipeline, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def run_pipeline():
    # Load and clean data
    df = load_and_clean_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df['Churn']
    
    # Train-test split before pipeline fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build and fit preprocessing pipeline
    pipeline = build_preprocessing_pipeline()
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    
    # Stratified K-Fold Cross Validation across multiple candidate models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost Baseline': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("--- Running 5-Fold Stratified Cross-Validation ---")
    for name, model in models.items():
        scores = cross_validate(model, X_train_trans, y_train, cv=cv, scoring=['recall', 'roc_auc'])
        print(f"{name} -> CV ROC-AUC: {scores['test_roc_auc'].mean():.4f} | CV Recall: {scores['test_recall'].mean():.4f}")
        
    # Hyperparameter Tuning on XGBoost
    print("\n--- Tuning XGBoost Hyperparameters ---")
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'scale_pos_weight': [1, 2.5, 3] # Addresses class imbalance
    }
    
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=8, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1)
    search.fit(X_train_trans, y_train)
    
    best_model = search.best_estimator_
    best_model.fit(X_train_trans, y_train)
    
    # Final evaluation on held-out test set
    y_pred = best_model.predict(X_test_trans)
    y_proba = best_model.predict_proba(X_test_trans)[:, 1]
    
    print(f"\nFinal Test Set ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Final Test Set Recall: {recall_score(y_test, y_pred):.4f}")
    
    # Ensure models directory exists
    import os
    os.makedirs('models', exist_ok=True)
    
    # Save artifacts in models/
    joblib.dump(best_model, 'models/model.pkl')
    joblib.dump(pipeline, 'models/pipeline.pkl')
    print("\nSuccessfully saved model.pkl and pipeline.pkl in models/")

if __name__ == '__main__':
    run_pipeline()