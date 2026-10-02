import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def load_and_clean_data(data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Loads raw CSV data and performs initial type coercions.
    NO STATISTICAL IMPUTATION OR SCALING IS PERFORMED HERE TO PREVENT LEAKAGE.
    """
    df = pd.read_csv(data_path)
    
    # Drop CustomerID as it's a unique non-predictive identifier
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Convert TotalCharges to numeric, setting invalid blank strings to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Convert target 'Churn' from Yes/No to 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def build_preprocessing_pipeline():
    """
    Returns an UNFITTED sklearn ColumnTransformer containing numeric and categorical pipelines.
    """
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor, numeric_features, categorical_features

def get_prepared_data(data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv", test_size=0.2, random_state=42):
    """
    Loads data and performs a stratified train/test split on RAW feature matrices.
    """
    df = load_and_clean_data(data_path)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test