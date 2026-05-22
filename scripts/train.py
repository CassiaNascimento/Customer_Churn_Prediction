from load_data import load_raw_data
from clean_data import clean_data
from split_data import split_data
from build_pipeline import build_pipeline
from evaluate_model import evaluate_model
from config import CHURN_MODEL_PATH, RANDOM_STATE
import joblib
import logging

logging.basicConfig(
    level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# Loading raw data
dataset=load_raw_data()

# Cleaning data
dataset_cleaned=clean_data(dataset)

# Splitting data
X_train, X_test, y_train, y_test=split_data(dataset_cleaned, random_state=RANDOM_STATE)

# Building Logistic Regression model
pipe=build_pipeline('Logistic Regression')

# Fit pipeline
fitted_pipe=pipe.fit(X_train, y_train)

# Evaluate model performance
metrics=evaluate_model(fitted_pipe,X_test,y_test)

# Save trained model
joblib.dump(fitted_pipe, CHURN_MODEL_PATH)

