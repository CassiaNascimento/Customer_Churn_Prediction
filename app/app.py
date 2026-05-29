import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.title("Customer churn prediction with Machine Learning")
st.write("This end-to-end machine learning project predicts the customer churn probability for a fictional telecommunications company. The raw data belongs to IBM Sample Data Sets and is available at [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). ")
st.write("Churn occurs when the customer cancels an ongoing contract. This machine learning project relies on classification models to predict, based on a set of 19 customer features, which ones are most likely to churn and guides the company onto developing retaining strategies.")
st.write("This web app was built and deployed with Streamlit :streamlit: and the python codes are available at my [GitHub page](https://github.com/CassiaNascimento/Customer_Churn_Prediction).")

tab1, tab2, tab3, tab4 = st.tabs([":open_file_folder: Take a look at the data", ":chart_with_upwards_trend: See the features trends", ":bookmark_tabs: Compare model perfomances", ":keyboard: Make new predictions"])

processed_data = pd.read_csv('../data/processed/cleaned_data.csv')
processed_data['SeniorCitizen'] = np.where(processed_data['SeniorCitizen']==0, "No", "Yes")
with tab1:
    st.write("Let's start taking a look at the cleaned dataset used in this project and extract some simple metrics.")
    num_rows = st.slider("Number of rows", 1, len(processed_data), 10)
    np.random.seed(42)

    data=[]
    for i in range(num_rows):
        data.append(processed_data.iloc[i,:])

    data=pd.DataFrame(data)

    number_of_churns=len(processed_data[processed_data["Churn"]=="Yes"])
    total_customers=len(processed_data)

    median_tenure_churn=processed_data[processed_data["Churn"]=="Yes"]["tenure"].median()
    median_monthly_charge_churn=processed_data[processed_data["Churn"]=="Yes"]["MonthlyCharges"].median()

    churn_rate=round(100*number_of_churns/total_customers,1)

    st.dataframe(data, width='stretch')

    col1, col2 = st.columns(2)
    col1.metric(label="Number of customers", value=len(processed_data), border=True)
    col2.metric(label="Churn Rate", value=f"{churn_rate}\%", border=True)

    col1, col2 = st.columns(2)
    col1.metric(label="Median Tenure", value=f"{round(processed_data['tenure'].median())}", border=True)
    col2.metric(label="Median Churn Tenure", value=f"{round(median_tenure_churn)}", border=True)

    col1, col2 = st.columns(2)
    col1.metric(label="Median Monthly Charge", value=f"$ {round(processed_data['MonthlyCharges'].median(),2)}", border=True)
    col2.metric(label="Median Monthly Charge Churn", value=f"$ {round(median_monthly_charge_churn,2)}", border=True)

    st.write("Now that we have a general picture of the data, we see that the customer that chooses to cancel the contract usually has higher monthly charges and a newer contract in comparison with the median customer.")

    st.write("Can we identify services that are highly correlated with churn? Let's explore the churn trends observed in each feature and construct our approach to model the customer churn probability.")

   #we know the dataset is imbalanced

with tab2:

    #st.write("Let's dive deeper into the trends observed in each feature, analising how they correlate with churn and investigate further our first realizations.")

    df_churn=processed_data[processed_data["Churn"]=="Yes"]
    df_no_churn=processed_data[processed_data["Churn"]=="No"]

    option = st.selectbox("Choose a numerical feature",("Duration of the contract", "Monthly charges per customer", "Total charges per customer"))

    #st.write("Indeed we should adress new customers retention and understand the higher fees this group oc customers")

    mapping = {"Duration of the contract":"tenure", "Monthly charges per customer":"MonthlyCharges", "Total charges per customer":"TotalCharges"}
    feat = mapping[option]

    num_bins = 20
    min_val = min(np.min(df_no_churn[feat]), np.min(df_churn[feat]))
    max_val = max(np.max(df_no_churn[feat]), np.max(df_churn[feat]))
    bin_size = (max_val - min_val) / num_bins

    xbins_setting = dict(start=min_val, end=max_val, size=bin_size)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_no_churn[feat],name="No Churn", xbins=xbins_setting))
    fig.add_trace(go.Histogram(x=df_churn[feat],name="Churn", xbins=xbins_setting))
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    st.plotly_chart(fig, width="stretch")

    option = st.selectbox("Let's take a look at the categorical features and understand a bit more about the patterns in customer behaviour:",("Is female or male?", "Is a senior citizen?", "Has a partner?", "Has dependents?", "Has phone service?", "Has multiple lines?", "Has internet service?", "Has online security?", "Has online backup?", "Has device protection?", "Has tech support?", "Has streaming TV?", "Has streaming movies?", "Type of contract?", "Receives paperless billing?", "Pays by which method?"))

    mapping = {"Is female or male?":"gender", "Is a senior citizen?": "SeniorCitizen", "Has a partner?":"Partner", "Has dependents?":"Dependents", "Has phone service?":"PhoneService", "Has multiple lines?":"MultipleLines", "Has internet service?":"InternetService", "Has online security?" :"OnlineSecurity", "Has online backup?":"OnlineBackup", "Has device protection?":"DeviceProtection", "Has tech support?":"TechSupport", "Has streaming TV?":"StreamingTV", "Has streaming movies?":"StreamingMovies", "Type of contract?":"Contract", "Receives paperless billing?":"PaperlessBilling", "Pays by which method?":"PaymentMethod"}

    feat = mapping[option]

    labels_churn = np.sort(df_churn[feat].unique())
    sizes_churn = [(df_churn[feat] == labels_churn[j]).sum() for j in range(len(labels_churn))]

    labels_not_churn = np.sort(df_no_churn[feat].unique())
    sizes_not_churn = [(df_no_churn[feat] == labels_not_churn[j]).sum() for j in range(len(labels_not_churn))]

    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],subplot_titles=["Churn", "Not Churn"])

    fig.add_trace(go.Pie(labels=labels_churn, values=sizes_churn, sort=False, hole=.3, name="Churn"),
              1, 1)
    fig.add_trace(go.Pie(labels=labels_not_churn, values=sizes_not_churn, sort=False, hole=.3, name="Not Churn"),
              1, 2)
    st.plotly_chart(fig, width="stretch")

with tab4:

    st.write("Now it is your turn to make new predictions. Choose the machine learning model, the metric and insert the customer features:")

    col1, col2 = st.columns(2)
    model = col1.radio(label="ML model",options=["XGBoost","LightGBM"],horizontal=True,label_visibility="hidden")

    metric = col2.radio(label="Metric",options=["Accuracy","Recall"],horizontal=True,label_visibility="hidden")

    option1 = st.selectbox("Is female or male?",("Female","Male"))
    option2 = st.selectbox("Is a senior citizen?",("Yes","No"))
    option3 = st.selectbox("Has a partner?",("Yes","No"))
    option4 = st.selectbox("Has dependents?",("Yes","No"))
    option6 = st.selectbox("Has phone service?",("Yes","No"))
    option7 = st.selectbox("Has multiple lines?",("Yes","No","No phone service"))
    option8 = st.selectbox("Has internet service?",("DSL", "Fiber optic","No"))
    option9 = st.selectbox("Has online security?",("Yes","No", "No internet service"))
    option10 = st.selectbox("Has online backup?",("Yes","No", "No internet service"))
    option11 = st.selectbox("Has device protection?",("Yes","No", "No internet service"))
    option12 = st.selectbox("Has tech support?",("Yes","No", "No internet service"))
    option13 = st.selectbox("Has streaming TV?",("Yes","No", "No internet service"))
    option14 = st.selectbox("Has streaming movies?",("Yes","No", "No internet service"))
    option15 = st.selectbox("What is the type of contract?",("Month-to-month", "One year", "Two year"))
    option16 = st.selectbox("Receives paperless billing?",("Yes","No"))
    option17 = st.selectbox("Pays by which method?",("Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"))
    option5 = st.slider("What is the contract duration?",0,75,25)
    option18 = st.slider("What are the Monthly Charges?",0,150,80)
    option19 = st.slider("What are the Total Charges?",0,8500,2000)

    st.subheader(f"Using {model} model and {metric.lower()} as metric, we find that the probability of this customer cancelling the contract is 30%.")















