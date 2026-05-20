from config import CHURN_MODEL_PATH
import joblib
import pandas as pd
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO, force=True)

def predict(sample_data):
    """ Predict churn for new customers
        1. Load fitted pipeline
        2. Obtain churn probability and prediction for new customer
    """
    fitted_pipeline=joblib.load(CHURN_MODEL_PATH)
    logging.info(f"Loaded fitted pipeline.")
    results=[]
    for item in sample_data:
        new = pd.DataFrame([item])
        y_prob=fitted_pipeline.predict_proba(new)[0,1]
        y_pred=fitted_pipeline.predict(new)[0]
        results.append({"prediction": int(y_pred),"probability": round(float(y_prob),4)})

    logging.info(f"Prediction finished.")
    return results

