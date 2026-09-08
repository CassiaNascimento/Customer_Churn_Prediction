# Customer Churn Prediction
This end-to-end machine learning project predicts the customer churn for a fictional telecommunications company. The raw data belongs to IBM Sample Data Sets and is available at [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Churn occurs when the customer cancels an ongoing contract. This ML project uses classification models to predict, based on a set of 19 features, which customers are likely to churn. This kind of prediction allows the fictional company to develop retaining strategies to avoid the churn. 

The project is structured as a modular Python package. Next steps include containerize the application using docker. 

## Streamlit Web App

This project includes an interactive web application built with Streamlit [].

## Quick Start (Run Locally)

1. Clone the repository from [GitHub](https://github.com/CassiaNascimento/Customer_Churn_Prediction)
2. Dowload the raw [dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/raw/` 
3. Create the Conda environment `conda env create -f environment.yml`
4. Activate the environment `conda activate churnenv`
5. Install the package from the project's root directory `pip install -e .`.
6. Navigate to the `app` folder and run the Streamlit application `streamlit run app.py`
7. Explore the Streamlit application and the included example notebooks.

## About the repository 
The dataset contains information on customer demographics, subscribed services and billing. A description of every feature is provided below. 

`customerID:` Unique customer identification number. \
`gender:` Whether the customer is a Male or a Female. \
`SeniorCitizen:` Whether the customer is a Senior Citizen. \
`Partner`: Whether the customer has a partner. \
`Dependents:` Whether the customer has dependents. \
`tenure:` Number of months the customer has been with the company. \
`PhoneService:` Whether the customer has phone service. Customers with phone service may also have `MultipleLines`. \
`MultipleLines:` Whether the customer has multiple phone lines. \
`InternetService:` Whether the customer has internet service and, if so, the type of internet service. Customers with internet service may also have `OnlineSecurity`, `OnlineBackup`, `DeviceProtection` and `TechSupport`, `StreamingTV` and `StreamingMovies`. \
`OnlineSecurity:` Whether the customer has a online security service. \
`OnlineBackup:` Whether the customer has a online backup service. \
`DeviceProtection:` Whether the customer has a device protection service. \
`TechSupport:` Whether the customer has a tech support service. \
`StreamingTV:` Whether the customer has a streaming TV service. \
`StreamingMovies:` Whether the customer has a streaming movies service. \
`Contract:` The type of contract: Month-to-month, One year, Two year. \
`PaperlessBilling:` Whether the customer has opted for paperless billing. \
`PaymentMethod:` The payment method chosen by the customer. \
`MonthlyCharges:` The monthly charge paid by the customer. \
`TotalCharges:` The total amount charged to the customer. \
`Churn:` Whether the customer churned.

During data cleaning we remove the `customerID` feature and exclude 11 customers with zero tenure and no recorded charges, corresponding to new customers. The remaining 7032 customers form our customer population, characterized by 16 categorical features and 3 numerical features. 

The repository is organized as follows:

- `src/` contains modules to load, clean, save and split the dataset, as well as engineer new features, build the training pipeline, evaluate classifiers performance and generate new predictions. 

- `notebooks/` contains an extensive Exploratory Data Analysis (EDA) with particular attention to identifying high churn risk groups. It also includes example notebooks for model training and for comparing classifier performance with and without hyperparameter optimization.

- `app/` contains the source code for the streamlit application.

- `models/` contains a .pkl file with the best performing classifier trained on this dataset. 

- `scripts/` contains example scripts for running data cleaning, model training, predictions and performance comparisons.

- `data/` contains the `raw/` and `processed/` folders, which are initially empty. The raw dataset must be downloaded and placed in the `raw/` folder. After running the data cleaning step, either in the notebook or through the scripts with the `save=True` flag, the cleaned dataset will be saved in the `processed/` folder.

- `reports/` contains .json files with the test set evaluation metrics, along with the ROC and AUC curves for each fitted model.

## Summary of findings

- Top Overall Performer: XGBoost achieved the highest Average Precision (PR-AUC) on the imbalanced dataset, yielding an ROC-AUC of 0.85 and a Recall rate of 0.79.

- Logistic Regression is a close second and secured top scores in 3 out of 6 evaluation metrics. After hyperparameter optimization using Optuna both models aggree on 93.8% of the test set predictions.
 
- To captured the majority of churners while controlling campaign expenses, targeted retention strategies based on decision threshold ranges are recommended.

- Additional strategies include offering incentives to migrate month-to-month contracts to one year contracts, and testing periods of complimentary services that are known to be associated with lower churn rates, as for example tech support.

- XGBoost is integrated into the Streamlit front-end for future predictions.