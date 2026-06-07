import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly
from predict import predict


st.title("Customer Churn Prediction with Machine Learning")
st.write("Identifying customers at risk of churning in advance allows companies to develop targeted retention strategies. This project applies supervised machine learning techniques to estimate customer churn probability for a fictional telecommunications company using the publicly available IBM Telco Customer Churn dataset.")
st.write("The project workflow includes data preprocessing, model training, hyperparameter tuning, and performance evaluation of multiple classification models. Given the class imbalance in the dataset, model performance is assessed using F1 score and ROC-AUC. The best performing model is integrated into this Streamlit application, allowing users to generate churn probability estimates for new customers.")
st.write("The source code is publicly available on my [GitHub page](https://github.com/CassiaNascimento/Customer_Churn_Prediction).")

tab1, tab2, tab3, tab4 = st.tabs([":open_file_folder: The dataset", ":chart_with_upwards_trend: Exploratory analysis", ":bookmark_tabs: Model perfomance comparison", ":keyboard: Make new predictions"])

processed_data = pd.read_csv('../data/processed/cleaned_data.csv')
processed_data['SeniorCitizen'] = np.where(processed_data['SeniorCitizen']==0, "No", "Yes")

df_churn=processed_data[processed_data["Churn"]=="Yes"]
df_no_churn=processed_data[processed_data["Churn"]=="No"]

with tab1:

    st.write("Below is the cleaned dataset used in this project, together with a set of key descriptive metrics.")
    num_rows = st.slider("Number of rows", 1, len(processed_data), 10)

    st.dataframe(processed_data.iloc[:num_rows,:], width='stretch')

    number_of_churns=len(df_churn)
    total_customers=len(processed_data)

    median_tenure_churn=df_churn["tenure"].median()
    median_tenure=processed_data['tenure'].median()

    median_monthly_charge_churn=df_churn["MonthlyCharges"].median()
    median_monthly_charge=processed_data["MonthlyCharges"].median()

    churn_rate=round(100*number_of_churns/total_customers,1)

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Customers", value=total_customers, border=True)
    col2.metric(label="Churned Customers",value=number_of_churns,border=True)
    col3.metric(label="Churn Rate", value=f"{churn_rate}\%", border=True)

    col1, col2 = st.columns(2)
    col1.metric(label="Median Tenure", value=f"{round(median_tenure)} months", border=True)
    col2.metric(label="Median Monthly Charge", value=f"$ {round(median_monthly_charge,2)}", border=True)

    col1, col2 = st.columns(2)
    col1.metric(label="Median Tenure (Churned Customers)", value=f"{round(median_tenure_churn)} months", delta=f"{round(median_tenure_churn-median_tenure)} months", border=True)
    col2.metric(label="Median Monthly Charge (Churned Customers)", value=f"$ {round(median_monthly_charge_churn,2)}", delta=f"${round(median_monthly_charge_churn-median_monthly_charge,2)}", border=True)

    st.write("The churn rate highlights the class imbalance in the dataset, while the remaining metrics suggest that customers who churn tend to have shorter tenures and higher monthly charges than the overall customer population.")

    st.write("In the next tab, we explore these differences in greater detail through an exploratory data analysis aimed at identifying churn patterns in customer attributes and subscribed services.")

with tab2:

    ######## First set of plots

    st.write("Choose a numerical feature and visualize how the churned customers differ from non-churned ones:")

    option1 = st.selectbox(" ",("Duration of the contract", "Monthly charges per customer", "Total charges per customer"), label_visibility="collapsed")

    mapping1 = {"Duration of the contract":"tenure", "Monthly charges per customer":"MonthlyCharges", "Total charges per customer":"TotalCharges"}
    feat1 = mapping1[option1]

    num_bins = 20
    min_val = min(np.min(df_no_churn[feat1]), np.min(df_churn[feat1]))-1
    max_val = max(np.max(df_no_churn[feat1]), np.max(df_churn[feat1]))+1
    bin_size = (max_val - min_val)/num_bins

    xbins_setting = {'start':min_val, 'end':max_val, 'size':bin_size}

    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=df_no_churn[feat1],name="No Churn", xbins=xbins_setting, marker_color='#5B8FF9'))
    fig1.add_trace(go.Histogram(x=df_churn[feat1],name="Churn", xbins=xbins_setting, marker_color='#E8684A'))
    fig1.update_layout(margin={'t':10, 'b':0},barmode='overlay',xaxis_title=f"{option1}", yaxis_title="# of customers")
    fig1.update_traces(opacity=0.8)
    st.plotly_chart(fig1, width="stretch", key="histograms")

    st.write("These graphics reinforce the initial metrics realizations, new customers retention is key to address contracts cancelation. Let's take a look at the categorical features and understand a bit more about the patterns in customer behaviour:")

    ######## Second set of plots

    option2 = st.selectbox(" ",("Is female or male?", "Is a senior citizen?", "Has a partner?", "Has dependents?", "Has phone service?", "Has multiple lines?", "Has internet service?", "Has online security?", "Has online backup?", "Has device protection?", "Has tech support?", "Has streaming TV?", "Has streaming movies?", "Type of contract?", "Receives paperless billing?", "Pays by which method?"), label_visibility="collapsed")

    mapping2 = {"Is female or male?":"gender", "Is a senior citizen?": "SeniorCitizen", "Has a partner?":"Partner", "Has dependents?":"Dependents", "Has phone service?":"PhoneService", "Has multiple lines?":"MultipleLines", "Has internet service?":"InternetService", "Has online security?" :"OnlineSecurity", "Has online backup?":"OnlineBackup", "Has device protection?":"DeviceProtection", "Has tech support?":"TechSupport", "Has streaming TV?":"StreamingTV", "Has streaming movies?":"StreamingMovies", "Type of contract?":"Contract", "Receives paperless billing?":"PaperlessBilling", "Pays by which method?":"PaymentMethod"}

    feat2 = mapping2[option2]

    categories=sorted(processed_data[feat2].unique(), key=len)

    if len(categories) ==4:
        fig2=make_subplots(rows=2, cols=2, specs=[[{'type':'domain'},{'type':'domain'}],[{'type':'domain'},{'type':'domain'}]], subplot_titles=categories)
        i=1
        k=1
        for cat in categories:
            labels=["No Churn","Churn"]
            sizes=[(df_no_churn[feat2]==cat).sum(), (df_churn[feat2]==cat).sum()]

            fig2.add_trace(go.Pie(labels=labels, values=sizes, sort=False, hole=.3, name=cat, marker=dict(colors=['#5B8FF9','#E8684A'])), k, i)
            if i%2==0:
                k+=1
                i=1
            else:
                i+=1
        fig2.update_layout(height=600,width=900)

    else:
        fig2=make_subplots(rows=1, cols=len(categories), specs=[[{'type':'domain'}]*len(categories)], subplot_titles=categories)

        i=1
        for cat in categories:
            labels=["No Churn","Churn"]
            sizes=[(df_no_churn[feat2]==cat).sum(), (df_churn[feat2]==cat).sum()]

            fig2.add_trace(go.Pie(labels=labels, values=sizes, sort=False, hole=.3, name=cat, marker=dict(colors=['#5B8FF9','#E8684A'])), 1, i)
            i+=1

    st.plotly_chart(fig2, width="stretch", key="pie_charts")
    st.write("Below we can analize the categorical features correlations with churn. ")

    ######## Third set of plots

    encoded_data=pd.get_dummies(processed_data)
    encoded_data=encoded_data.drop(columns=["PaperlessBilling_No","Dependents_No","Partner_No","SeniorCitizen_No","gender_Female","PhoneService_No","Churn_No"])
    corr_w_churn=encoded_data.corr()["Churn_Yes"].sort_values(ascending=True).reset_index()
    corr_w_churn.columns = ['variable', 'correlation']
    corr_w_churn['correlation']=round(corr_w_churn['correlation'],3)

    custom_scale = [[0.0, "#5B8FF9"],[0.5, "#FFFFFF"],[1.0, "#E8684A"]]

    fig3 = px.bar(corr_w_churn, x='correlation', y='variable', orientation='h', color='correlation',  labels={"correlation": "Corr.:"}, color_continuous_scale=custom_scale, color_continuous_midpoint=0)

    fig3.update_layout(margin={'t':10, 'b':10}, height=800,xaxis_title=None, yaxis_title=None)
    fig3.update_xaxes(visible=False)

    st.plotly_chart(fig3, width="stretch", key="correlations_bar_plot")

with tab4:

    st.write("Now it is your turn to make new predictions. Choose the machine learning model, the metric and insert the new customer features:")

    with st.form("my_form"):
        col1, col2 = st.columns(2)
        #model = col1.radio(label="Model:",options=["XGBoost","LightGBM"],horizontal=True)
        #metric = col2.radio(label="Metric:",options=["Accuracy","F1 Score"],horizontal=True)

        model = col1.radio(label="Model:",options=["Logistic Regression"],horizontal=True)
        metric = col2.radio(label="Metric:",options=["F1 Score"],horizontal=True)

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

        mapping={"Yes":1,"No":0}

        st.form_submit_button("Predict")

        sample_customer = {
            "gender": option1,
            "SeniorCitizen": mapping[option2],
            "Partner": option3,
            "Dependents": option4,
            "tenure": option5,
            "PhoneService": option6,
            "MultipleLines": option7,
            "InternetService": option8,
            "OnlineSecurity": option9,
            "OnlineBackup": option10,
            "DeviceProtection": option11,
            "TechSupport": option12,
            "StreamingTV": option13,
            "StreamingMovies": option14,
            "Contract": option15,
            "PaperlessBilling": option16,
            "PaymentMethod": option17,
            "MonthlyCharges": option18,
            "TotalCharges":option19}

    results=predict(sample_data=[sample_customer], model=model)

    prob=round(results[0]["probability"]*100,2)

    st.subheader(f"The churning probability for this customer is: {prob}%.")


# FEATURE ENGENEERING
# skeweness could lead to a lot of false negatives.
# Lowering the threshold from 0.5 generally improves Recall, capturing more actual churners — which is the priority in a business context where missing a churner is more costly than a false alarm.

# TAB 3:
# CONFUSION MATRIX
# SHAP FEATURE IMPORTANCE
# ROC curve??




