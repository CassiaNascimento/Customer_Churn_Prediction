from load_data import load_raw_data
from clean_data import clean_data
import os
import logging

logging.basicConfig(
    level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# Loading raw data
dataset=load_raw_data()

# Cleaning data
dataset_cleaned=clean_data(dataset)

if not os.path.exists("../data/processed"):
    os.makedirs("../data/processed")

# Saving the cleaned data 
dataset_cleaned.to_csv('../data/processed/cleaned_data.csv')