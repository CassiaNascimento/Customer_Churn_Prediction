from config import CHURN_MODEL_PATH
import joblib
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def predict(sample_data: list[dict[str, str|int|float]], model: str) -> list[dict[str, int|float]]:
    """ Predict churn for new customers

        Steps:
            1. Transform input into DataFrame
            2. Obtain churn probability and prediction for new customer

        Args:
            sample_data: List of dictionaries containing new customer features.
        Returns:
            List of dictionaries with customer churn probabilities and predictions.

        To do:
            1. Create validation schema
            2. Create Customer Class.
    """

    if model=="Logistic Regression":
        try:
            fitted_pipeline=joblib.load(CHURN_MODEL_PATH)
            logger.info("Default pipeline sucessfully loaded.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model not found at {CHURN_MODEL_PATH}. Please check the directory.")

    input_df=pd.DataFrame(sample_data)

    y_prob=fitted_pipeline.predict_proba(input_df)[:,1]
    y_pred=fitted_pipeline.predict(input_df)

    logger.info("Prediction completed for %d samples.", len(sample_data))

    return [{'prediction': int(pred),'probability': round(float(prob), 4),} for pred, prob in zip(y_pred, y_prob)]

