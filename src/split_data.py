from config import TARGET, RANDOM_STATE
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

def split_data(dataset: pd.DataFrame, test_size: float=0.2, random_state: int=RANDOM_STATE)-> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ Split the cleaned data

        Split cleaned data into train and test sets with proportion 0.8/0.2 as default.

        Args:
            dataset: Cleaned dataset.
            test_size: Size of the test set. Default is 0.2.
            random_state: Random state to ensure reproducibility.
        Returns:
            Features train and test sets as DataFrames and target train and test sets as Series.
    """
    X=dataset.drop(columns=[TARGET], axis=1)
    y=dataset[TARGET].map({'No':0, 'Yes':1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    logger.info("Data split into train/test set with %.1f/%.1f proportion and random_state=%d.",1-test_size, test_size, random_state)

    return X_train, X_test, y_train, y_test

