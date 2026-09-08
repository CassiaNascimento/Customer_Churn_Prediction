from load_data import load_raw_data
from clean_data import clean_data
import os
import logging

logging.basicConfig(
    level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# Loading raw data
dataset=load_raw_data()

# Cleaning data and saving if save=True
dataset_cleaned=clean_data(dataset, save=False)

print(dataset_cleaned)