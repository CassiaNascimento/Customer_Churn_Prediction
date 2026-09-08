from load_data import load_raw_data
from clean_data import clean_data
from split_data import split_data
from build_pipeline import objective,build_pipeline
from feature_engineering import build_new_features
from evaluate_model import evaluate_model, view_ROC_curve, view_PR_curve
from config import MODEL_PATH, RANDOM_STATE, REPORT_DIR
import matplotlib.pyplot as plt
import optuna
import joblib
import logging
import json

logging.basicConfig(
    level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# Available models:
#
# XGBoost
# Logistic Regression
# K-Nearest Neighbors
# Support Vector Machine
# Decision Tree
# Random Forest

MODEL_NAME = "Support Vector Machine"
USE_OPTUNA = True
SAVE_METRICS = True
N_TRIALS = 100 
threshold=0.5

# Loading raw data
dataset=load_raw_data()

# Cleaning data
dataset_cleaned=clean_data(dataset, save=False)

# Adding new features
#dataset_cleaned=build_new_features(dataset_cleaned, new_features=["NumberServices","CustomerLevel","MonthlyChargePerService","AverageMonthlyCharge"])

# Splitting data
X_train, X_test, y_train, y_test=split_data(dataset_cleaned, random_state=RANDOM_STATE)

best_trial=None

if USE_OPTUNA == True:
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study=optuna.create_study(direction="maximize",sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train, model=MODEL_NAME),n_trials=N_TRIALS)
    best_trial=study.best_trial
    print("Best score:", round(best_trial.value,3))
    print("Best params:", best_trial.params)

# Building pipeline
pipeline=build_pipeline(trial=best_trial, model=MODEL_NAME)

# Fitting pipeline
fitted_pipe=pipeline.fit(X_train, y_train)

# Evaluating model performance
metrics,report=evaluate_model(fitted_pipe,X_test,y_test,threshold=threshold)

print("\n"+"="*60)
print(f"Model performance on test data ({MODEL_NAME}):")
print("="*60)
for k, v in metrics.items():
    print(f"- {k:<20}: {v}")
print("="*60)
print(f"Sklearn classification report ({MODEL_NAME}):")
print("=" * 60)
print("\n" + report)
print("="*60)

# Save figures
if USE_OPTUNA==True:
    view_ROC_curve(fitted_pipe,X_test,y_test,save=True, save_path=REPORT_DIR/f"figures/ROC_{MODEL_NAME.lower().replace(' ', '_')}_tuned.png")
    view_PR_curve(fitted_pipe,X_test,y_test,save=True, save_path=REPORT_DIR/f"figures/PR_{MODEL_NAME.lower().replace(' ', '_')}_tuned.png")
else:
    view_ROC_curve(fitted_pipe,X_test,y_test,save=True, save_path=REPORT_DIR/f"figures/ROC_{MODEL_NAME.lower().replace(' ', '_')}.png")
    view_PR_curve(fitted_pipe,X_test,y_test,save=True, save_path=REPORT_DIR/f"figures/PR_{MODEL_NAME.lower().replace(' ', '_')}.png")

# Save trained model
if USE_OPTUNA==True:
    file_name = f"{MODEL_NAME.lower().replace(' ', '_')}_pipeline_tuned.pkl"
else:
    file_name = f"{MODEL_NAME.lower().replace(' ', '_')}_pipeline.pkl"
joblib.dump(fitted_pipe, MODEL_PATH/file_name)

# Save metrics
if SAVE_METRICS==True:
    th_suffix = f"th_{str(threshold).replace('.', '')}"
    if USE_OPTUNA==True:
        json_path = REPORT_DIR / f"{MODEL_NAME.lower().replace(' ', '_')}_metrics_tuned_{th_suffix}.json"
    else:
        json_path = REPORT_DIR / f"{MODEL_NAME.lower().replace(' ', '_')}_metrics_{th_suffix}.json"

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Results successfully saved.")