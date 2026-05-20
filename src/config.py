from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
CHURN_MODEL_PATH = PROJECT_ROOT/'models/churn_model.pkl'

TARGET = 'Churn'
BINARY_FEATURES=['gender','SeniorCitizen','Partner','Dependents','PhoneService','PaperlessBilling']
CATEGORICAL_FEATURES=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies', 'Contract', 'PaymentMethod']
NUMERICAL_FEATURES=['tenure','MonthlyCharges','TotalCharges']

ORDERED_FEATURES=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies', 'Contract','PaperlessBilling', 'PaymentMethod','MonthlyCharges','TotalCharges']
