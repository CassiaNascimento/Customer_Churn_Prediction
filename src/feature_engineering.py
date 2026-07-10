import pandas as pd
import numpy as np
import plotly.express as px
from config import  SERVICE_COLUMNS
import logging

logger = logging.getLogger(__name__)

def build_new_features(dataset: pd.DataFrame)->pd.DataFrame:
    """ Build new features

        Args:
            dataset: Cleaned dataset.
        Returns:
            Original dataframe with new feature added
    """
    dataset = dataset.copy()

    for col in ['tenure']+SERVICE_COLUMNS:
        if col not in dataset.columns:
            raise ValueError(f"Column {col} not found.")

    mapping={"Yes":1, "Fiber optic":1, "DSL":1, "No":0, "No internet service":0}

    n_services=np.zeros(len(dataset))

    for col in SERVICE_COLUMNS:
        n_services+=dataset.loc[:,col].map(mapping).to_numpy()

    dataset['NumberServices']=n_services
    dataset['NumberServices']=dataset['NumberServices'].astype('int64')

    dataset['monthly_charge_per_service']=dataset['MonthlyCharges']/n_services

    dataset['is_month_to_month_and_fiber']=((dataset['Contract']=='Month-to-month') & (dataset['InternetService']=='Fiber optic')).astype(int)

    dataset['has_security_or_support']=((dataset['OnlineSecurity']=='Yes') & (dataset['TechSupport']=='Yes')).astype(int)

    dataset['average_monthly_spend']=dataset['TotalCharges']/dataset['tenure']

    dataset['CustomerLevel']=pd.cut(dataset['tenure'],bins=[0, 6, 12, 24, 48 ,100],labels=[0,1,2,3,4]).astype('int64')

    rows_with_nan=dataset.isna().any(axis=1).sum()

    logger.info("Added NumberServices and CustomerLevel features. Current rows with NaN: %d.", rows_with_nan)

    return dataset
