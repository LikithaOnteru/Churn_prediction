# 📊 Customer Churn Prediction & Retention Intelligence System

An end-to-end Machine Learning pipeline and interactive intelligence dashboard designed to identify high-risk subscription churners, explain model decisions using SHAP (SHapley Additive exPlanations), and provide automated, rule-based retention strategies for customer success teams.

---

## 📌 Executive Summary & Key Results

In subscription business models, the financial cost of losing a customer (**False Negative**) far outweighs the minor cost of offering a retention discount (**False Positive**). This system prioritizes **Recall** and **ROC-AUC** to maximize customer retention while eliminating data leakage through modular `scikit-learn` pipeline design.

* **CV ROC-AUC:** `0.8461` (Logistic Regression baseline) / `0.8228` (XGBoost baseline)
* **Final Test Set ROC-AUC:** **`0.8433`**
* **Final Test Set Recall:** **`0.8021`** *(Captures 80.2% of all at-risk churners)*

---

## 🏗 System Architecture

```text
                               ┌─────────────────────────────────────────┐
                               │     Telco Customer Churn Dataset        │
                               │  (WA_Fn-UseC_-Telco-Customer-Churn.csv) │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │        src/preprocessing.py             │
                               │  - Clean TotalCharges (Numeric Coerce)  │
                               │  - Train/Test Split (Stratified 80/20)  │
                               │  - ColumnTransformer (Impute/Scale/OHE) │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │           src/train.py                  │
                               │  - Stratified 5-Fold Cross-Validation   │
                               │  - XGBoost RandomizedSearchCV Tuning    │
                               │  - Save model.pkl & pipeline.pkl        │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │           models/ Artifacts             │
                               │     (model.pkl  |  pipeline.pkl)        │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │              app.py                     │
                               │  - Streamlit Interactive Web Interface  │
                               │  - 4-Tier Risk Classification           │
                               │  - Instance-Level SHAP Explainability   │
                               │  - src/recommendations.py Business Rules│
                               └─────────────────────────────────────────┘
