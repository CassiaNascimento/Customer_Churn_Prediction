from config import TARGET
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO, force=True)

def split_data(dataset, test_size=0.2, random_state=0):
    """ Split data
        1. Split data in train and test sets with proportion 0.8/0.2 (default)
    """
    X=dataset.drop(TARGET, axis=1)
    y=dataset[TARGET].map({'No':0, 'Yes':1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    logging.info(f"Splitted the data into train/test set with {1-test_size}/{test_size} proportion and random_state={random_state}.")

    return X_train, X_test, y_train, y_test

