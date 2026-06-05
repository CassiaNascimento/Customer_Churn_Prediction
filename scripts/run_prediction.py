from predict import predict
import logging

logging.basicConfig(
    level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

sample_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 7,
    "PhoneService": "No",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 30,
    "TotalCharges": 340.
}

results=predict([sample_customer,sample_customer,sample_customer,sample_customer])

print(results)

# tenure, monthly charges and total charges: the data must obbey a range
