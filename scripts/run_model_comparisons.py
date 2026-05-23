from load_data import load_raw_data
from clean_data import clean_data
from split_data import split_data
from build_pipeline import build_pipeline
from evaluate_model import evaluate_model
from config import CHURN_MODEL_PATH, RANDOM_STATE, MODEL_PATH
import joblib
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# Loading raw data
dataset=load_raw_data()

# Cleaning data
dataset_cleaned=clean_data(dataset)

# Splitting data
X_train, X_test, y_train, y_test=split_data(dataset_cleaned, random_state=RANDOM_STATE)

# Building Logistic Regression model
pipe_LR=build_pipeline('Logistic Regression')
fitted_pipe_LR=pipe_LR.fit(X_train, y_train)

# Building K-Nearest Neighbors
pipe_KNN=build_pipeline('K-Nearest Neighbors')
fitted_pipe_KNN=pipe_KNN.fit(X_train, y_train)

# Building Support Vector Machine model
pipe_SVM=build_pipeline('Support Vector Machine')
fitted_pipe_SVM=pipe_SVM.fit(X_train, y_train)

# Building Decision Tree model
pipe_DT=build_pipeline('Decision Tree')
fitted_pipe_DT=pipe_DT.fit(X_train, y_train)

# Building Random Forest model
pipe_RF=build_pipeline('Random Forest')
fitted_pipe_RF=pipe_RF.fit(X_train, y_train)

# Evaluate model performance
metrics_LR=evaluate_model(fitted_pipe_LR,X_test,y_test)
print(metrics_LR['report'])

metrics_KNN=evaluate_model(fitted_pipe_KNN,X_test,y_test)
print(metrics_KNN['report'])

metrics_SVM=evaluate_model(fitted_pipe_SVM,X_test,y_test)
print(metrics_SVM['report'])

metrics_DT=evaluate_model(fitted_pipe_DT,X_test,y_test)
print(metrics_DT['report'])

metrics_RF=evaluate_model(fitted_pipe_RF,X_test,y_test)
print(metrics_RF['report'])

# Save trained model
path_LR = Path(MODEL_PATH/"Logistic_Regression")
path_LR.mkdir(exist_ok=True)
joblib.dump(fitted_pipe_LR, path_LR/"model_LR.pkl")

path_KNN = Path(MODEL_PATH/"K_Nearest_Neighbors")
path_KNN.mkdir(exist_ok=True)
joblib.dump(fitted_pipe_KNN, path_KNN/"model_KNN.pkl")

path_SVM = Path(MODEL_PATH/"Support_Vector_Machine")
path_SVM.mkdir(exist_ok=True)
joblib.dump(fitted_pipe_SVM, path_SVM/"model_SVM.pkl")

path_DT = Path(MODEL_PATH/"Decision_Tree")
path_DT.mkdir(exist_ok=True)
joblib.dump(fitted_pipe_DT, path_DT/"model_DT.pkl")

path_RF = Path(MODEL_PATH/"Random_Forest")
path_RF.mkdir(exist_ok=True)
joblib.dump(fitted_pipe_RF, path_RF/"model_RF.pkl")

