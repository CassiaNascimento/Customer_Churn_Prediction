from config import CHURN_MODEL_PATH, MODEL_PATH
import joblib
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_pipeline(model: str):
    file_name = f"{model.lower().replace(' ', '_')}_pipeline_tuned.pkl"
    filepath = MODEL_PATH / file_name
    if not filepath.exists():
        raise FileNotFoundError(f"Model '{model}' not found at {filepath}.")
    pipeline = joblib.load(filepath)
    logger.info(f"Pipeline '{model}' successfully loaded.")
    return pipeline

def predict(sample_data: list[dict[str, str|int|float]], model: str, threshold: float = 0.5) -> list[dict[str, int|float]]:
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

    fitted_pipeline=load_pipeline(model)
    
    input_df=pd.DataFrame(sample_data)

    y_prob=fitted_pipeline.predict_proba(input_df)[:,1]
    y_pred=(y_prob>=threshold).astype(int)
    
    logger.info("Prediction completed for %d samples.", len(sample_data))

    if "customerID" in input_df.columns:
        return [{'customer_ID':input_df["customerID"][index],'threshold': float(threshold),'prediction': int(pred),'probability': round(float(prob), 4)} for index, (pred, prob) in enumerate(zip(y_pred, y_prob))]
    return [{'threshold': float(threshold),'prediction': int(pred),'probability': round(float(prob), 4)} for index, (pred, prob) in enumerate(zip(y_pred, y_prob))]

