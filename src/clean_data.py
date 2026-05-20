import pandas as pd
import numpy as np
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO, force=True)

def clean_data(dataset):
    """ Clean the raw data:
        1. Drop the ID column
        2. Assign numeric type to Total Charges column
        3. Drop NaN Total Charges values, which correspond to new customers
        4. Count the remaining NaN occurences
    """
    dataset=dataset.drop(columns=['customerID'])
    logging.info(f"Dropped CustomerID column.")

    dataset['TotalCharges']=pd.to_numeric(dataset['TotalCharges'], errors='coerce')
    logging.info(f"Corrected data types.")

    nan_count_All=dataset.isna().any(axis=1).sum()
    logging.info(f"Total of {nan_count_All} NaNs.")

    nan_count_TotalCharges=dataset['TotalCharges'].isna().sum()
    dataset=dataset.dropna(subset=['TotalCharges'], ignore_index=True)
    logging.info(f"Dropped {nan_count_TotalCharges} rows with NaN TotalCharges.")

    nan_count_All=dataset.isna().any(axis=1).sum()
    logging.info(f"Total of {nan_count_All} NaNs remain.")

    return dataset.copy()


