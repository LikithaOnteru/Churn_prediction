import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
TARGET = 'Churn'

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # Convert TotalCharges to numeric, coercing blank spaces safely
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    # Impute missing TotalCharges values with median
   # New line (clean Pandas Copy-on-Write syntax)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median()) 
    # Treat SeniorCitizen as string category for uniform encoding
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    # Map target
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({'Yes': 1, 'No': 0})
    return df

def build_preprocessing_pipeline():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, NUMERICAL_FEATURES),
        ('cat', cat_pipeline, CATEGORICAL_FEATURES)
    ])
    
    return preprocessor