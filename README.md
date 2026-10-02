# Customer Churn Prediction & Retention Intelligence

## Overview
This repository provides an end-to-end Machine Learning system that predicts customer churn probability and translates raw predictions into explainable, rule-based retention recommendations. Built with Scikit-Learn, XGBoost, SHAP, and Streamlit, it optimizes decision boundaries to maximize high-cost churn detection in customer retention workflows.

## Problem Statement
Customer churn directly impacts recurring subscription revenue. Identifying high-risk customers early enables customer success teams to deliver targeted, cost-effective retention offers. Standard classification models using default decision thresholds ($\tau = 0.500$) often fail to capture sufficient churners due to class imbalance and unequal misclassification costs.

## Key Results
Evaluated on an untouched holdout test set (20% sample, 1,409 customers):

| Metric | Result |
| :--- | :--- |
| **ROC-AUC** | **0.8459** |
| **Accuracy** | **0.6799** |
| **Precision** | **0.4483** |
| **Recall** | **0.8930** |
| **F1-score** | **0.5970** |
| **F2-score** | **0.7452** |
| **Optimized Decision Threshold ($\tau$)** | **0.1836** |

### Decision Threshold Optimization Comparison
Threshold calibration was selected using Training Out-Of-Fold (OOF) predictions to maximize the $F_2$-score ($C_{FN} \gg C_{FP}$):

| Decision Threshold ($\tau$) | Precision | Recall | F2-Score | Operational Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Default ($\tau = 0.500$)** | 68.83% | 45.45% | 0.4877 | Misses over 54% of actual churners |
| **Optimized ($\tau = 0.1836$)** | 44.83% | **89.30%** | **0.7452** | **Captures 89.3% of churn risk early** |

> **Metric Clarity Note:** At the optimized decision threshold of 0.1836, the model achieved **89.3% recall** on the untouched test set.

## Machine Learning Pipeline
1. **Data Ingestion:** Reads Telco customer records and enforces numeric type casting on `TotalCharges`.
2. **Stratified Split:** 80/20 train/test split preserving target class ratio.
3. **Preprocessing:** `ColumnTransformer` with `OneHotEncoder` for categorical variables and `SimpleImputer` + `StandardScaler` for numeric variables inside a single `Pipeline`.
4. **Model Tuning:** `RandomizedSearchCV` cross-validation on `XGBClassifier` with 5-fold `StratifiedKFold`.
5. **Threshold Calibration:** Out-of-fold probability optimization selecting $\tau = 0.1836$ based on training data only (zero test set leakage).
6. **Persistence:** Single canonical artifact saved to `models/pipeline.pkl`.

## Explainability
Local feature attributions are computed using `shap.TreeExplainer` on the preprocessed feature space. SHAP highlights individual feature contributions driving churn probability up or down for a given customer profile.

## Retention Recommendations
The system passes model predictions into a prescriptive `ChurnRetentionEngine`. Based on risk tiers (*Low*, *Medium*, *High*, *Critical Risk*) and customer service configurations, the engine outputs rule-based retention actions.
*Note: Recommendations are deterministic business rules designed for operational decision support and are not statistical claims of causal intervention efficacy.*

## System Architecture
```text
Customer Input Data
        │
        ▼
scikit-learn Pipeline (preprocessing + XGBoost)
        │
        ▼
Churn Probability Score (0.0 to 1.0)
        │
        ▼
Optimized Decision Threshold (τ = 0.1836)
        │
        ▼
Risk Classification Tier
        │
  ┌─────┴─────────────────────┐
  ▼                           ▼
SHAP Feature Importance    Rule-Based Retention Actions