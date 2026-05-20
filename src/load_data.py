import pandas as pd
from config import RAW_DATA_DIR
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO, force=True)

def load_raw_data(filename=RAW_DATA_DIR/"WA_Fn-UseC_-Telco-Customer-Churn.csv", encoding="utf-8"):
    """ Load the raw data
        1. IBM sample data available at https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

        To do:
        1. Check the file exists
        2. Validate schema/shape
    """
    dataset=pd.read_csv(filename)
    logging.info(f"Loaded raw data.")

    return dataset.copy()
