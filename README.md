# Customer Churn Prediction
This ready to deploy machine learning project predicts the customer churn for a fictional telecommunications company. The raw data belongs to IBM Sample Data Sets and is available at [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Churn occurs when the customer cancels an ongoing contract. This ML project utilises classification models to predict, based on a set of 20 features, which customers are likely to churn. This kind of prediction allows the fictional company to develop retaining strategies to avoid the churn. 

The project is organized as a python module and is ready for deployment. Next steps include creating a docker image and deploying it on the cloud. A streamlit front-end is in progress.

## About the data set: 
`customerID: ` Unique customer identification number \
`gender: ` Whether the customer is a Male or a Female \
`SeniorCitizen: ` Whether the customer is a Senior Citizen \
`Partner`: Whether the customer has a partner \
`Dependents: ` Whether the customer has dependents \
`tenure: ` The duration in months of the customer contract \
`PhoneService: ` Whether the customer has phone service (for which the customer may also have `MultipleLines`) \
`InternetService: ` Whether the customer has phone service (for which the customer may also have `OnlineSecurity`, `OnlineBackup`, `DeviceProtection` and `TechSupport`) \
`StreamingTV: ` Whether the customer has streaming TV service \
`StreamingMovies: ` Whether the customer has streaming movies service \
`Contract: ` The recurrance of the contract, i. e. Monthly, Yearly, Bi-yearly \
`PaperlessBilling: ` Whether the customer receives the billing in paper \
`PaymentMethod: ` The payment method choosen by the customer \
`MonthlyCharges: ` The monthly charge paid by the customer \
`TotalCharges: ` The total amount of charges paid by the customer \
`Churn: ` Whether the customer cancelled their contract last month

