import pandas as pd
import numpy as np
import plotly.express as px
from config import  SERVICE_COLUMNS
import logging

logger = logging.getLogger(__name__)

def build_new_features(dataset: pd.DataFrame, new_features: list=["NumberServices", "CustomerLevel", "MonthlyChargePerService", "AverageMonthlyCharge"])->pd.DataFrame:
    """ Build new features

        Args:
            dataset: Cleaned dataset.
        Returns:
            Original dataframe with new features added
    """
    dataset=dataset.copy()
    new_features_added=[]

    for col in ['tenure']+SERVICE_COLUMNS:
        if col not in dataset.columns:
            raise ValueError(f"Column {col} not found.")

    mapping={"Yes":1, "Fiber optic":1, "DSL":1, "No":0, "No internet service":0}

    n_services=np.zeros(len(dataset))

    for col in SERVICE_COLUMNS:
        n_services+=dataset.loc[:,col].map(mapping).to_numpy()

    # Number of services subscribed
    if "NumberServices" in new_features:
        dataset['NumberServices']=n_services
        dataset['NumberServices']=dataset['NumberServices'].astype('int64')
        new_features_added.append("NumberServices")

    # Customer seniority rank
    if "CustomerLevel" in new_features:
        dataset['CustomerLevel']=pd.cut(dataset['tenure'],bins=[-np.inf, 11, 23, 35, 47, np.inf],labels=[0,1,2,3,4]).astype('int64')
        new_features_added.append("CustomerLevel")

    # Monthly Charges per number of services subscribed
    if "MonthlyChargePerService" in new_features:
        dataset['MonthlyChargePerService']=dataset['MonthlyCharges']/n_services
        new_features_added.append("MonthlyChargePerService")

    # Charges averaged per tenure
    if "AverageMonthlyCharge" in new_features:
        dataset['AverageMonthlyCharge']=dataset['TotalCharges']/dataset['tenure']
        new_features_added.append("AverageMonthlyCharge")

    rows_with_nan=dataset.isna().any(axis=1).sum()

    logger.info("Added the following new features: %s. Current rows with NaN: %d.",new_features_added, rows_with_nan)

    return dataset
