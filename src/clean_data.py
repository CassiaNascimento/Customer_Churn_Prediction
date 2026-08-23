import pandas as pd
from config import PROCESSED_DATA_DIR
import logging

logger = logging.getLogger(__name__)

def clean_data(dataset: pd.DataFrame, save: bool =False) -> pd.DataFrame:
    """ Clean the raw data

        Steps:
            1. Drop the ID column
            2. Assign numeric type to Total Charges column
            3. Drop NaN Total Charges values, which correspond to new customers
            4. Count the remaining rows with NaN occurences

        Args:
            dataset: Raw dataset.
        Returns:
            Cleaned dataset as DataFrame.
    """
    dataset=dataset.copy()
    dataset=dataset.drop(columns=['customerID'])
    logger.info("Dropped customerID.")

    dataset['TotalCharges']=pd.to_numeric(dataset['TotalCharges'], errors='coerce')
    logger.info("Converted TotalCharges to numeric.")

    rows_with_nan=dataset.isna().any(axis=1).sum()
    logger.info("Rows containing NaN before cleaning: %d", rows_with_nan)

    dropped=dataset['TotalCharges'].isna().sum()
    dataset=dataset.dropna(subset=['TotalCharges'], ignore_index=True)
    remaining_rows_with_nan=dataset.isna().any(axis=1).sum()
    logger.info("Removed %d rows with invalid TotalCharges. Remaining rows with NaN: %d.", dropped, remaining_rows_with_nan)

    if save is True:
        if PROCESSED_DATA_DIR.exists():
            dataset.to_csv(PROCESSED_DATA_DIR/"cleaned_data.csv",index=False)
            logger.info("Saved cleaned data at: %s.", PROCESSED_DATA_DIR)
        else:
            raise FileNotFoundError(f"Directory not found: {PROCESSED_DATA_DIR}")

    return dataset


