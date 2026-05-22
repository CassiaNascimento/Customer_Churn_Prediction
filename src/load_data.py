import pandas as pd
from pathlib import Path
from config import RAW_DATA_DIR, TARGET
import logging

logger = logging.getLogger(__name__)

def load_raw_data(filename: Path=RAW_DATA_DIR/"WA_Fn-UseC_-Telco-Customer-Churn.csv", encoding: str="utf-8") -> pd.DataFrame:
    """ Load the raw data

        IBM sample data available at https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

        Args:
            filename: Path to CSV file.
            encoding: File encoding.
        Returns:
            Loaded dataset as DataFrame.
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If dataset is empty or if the target column is missing.

        To do:
        1. Validate schema (check for required columns, column types, allowed values, duplicates)
    """

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    dataset=pd.read_csv(filename, encoding=encoding)

    if dataset.empty:
        raise ValueError("Dataset is empty.")
    if TARGET not in dataset.columns:
        raise ValueError("Target column not found.")

    logger.info("Raw dataset loaded.")

    return dataset
