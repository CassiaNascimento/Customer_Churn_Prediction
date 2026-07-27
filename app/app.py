# streamlit app
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from predict import predict
from build_pipeline import build_preprocessor
from config import RANDOM_STATE, NUMERICAL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET
from sklearn.feature_selection import SelectKBest,f_classif, chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

st.title("Customer Churn Prediction with Machine Learning")
st.write("Identifying customers at risk of churning in advance allows companies to develop targeted retention strategies. This project applies supervised machine learning techniques to estimate customer churn probability for a fictional telecommunications company using the publicly available IBM Telco Customer Churn dataset.")
st.write("The project workflow includes data preprocessing, feature engineering, model training, hyperparameter tuning, and performance evaluation of multiple classification models. Given the class imbalance in the dataset, model performance is assessed using F1 score and ROC-AUC. The best performing model is integrated into this Streamlit application, allowing users to generate churn probability estimates for new customers.")
st.write("The source code is publicly available on my [GitHub page](https://github.com/CassiaNascimento/Customer_Churn_Prediction).")

tab1, tab2, tab3, tab4 = st.tabs([":open_file_folder: The dataset", ":chart_with_upwards_trend: Exploratory analysis", ":bookmark_tabs: Model perfomance report", ":keyboard: Make new predictions"])

processed_data=pd.read_csv('../data/processed/cleaned_data.csv')
processed_data['SeniorCitizen']=np.where(processed_data['SeniorCitizen']==0, "No", "Yes")

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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Customers", value=total_customers, border=True)
    col2.metric(label="Churned Customers",value=number_of_churns,border=True)
    col3.metric(label="Churn Rate", value=f"{churn_rate}\%", border=True)
    col4.metric(label="Revenue Attrition Rate", value="30.5%",border=True)

    col1, col2 = st.columns(2)
    col1.metric(label="Median Tenure", value=f"{round(median_tenure)} months", border=True)
    col2.metric(label="Median Monthly Charge", value=f"$ {round(median_monthly_charge,2)}", border=True)

    col1, col2 = st.columns(2)
    col1.metric(label="Median Tenure (Churned Customers)", value=f"{round(median_tenure_churn)} months", delta=f"{round(median_tenure_churn-median_tenure)} months", border=True)
    col2.metric(label="Median Monthly Charge (Churned Customers)", value=f"$ {round(median_monthly_charge_churn,2)}", delta=f"${round(median_monthly_charge_churn-median_monthly_charge,2)}", border=True)

    st.write("The churn rate highlights the class imbalance in the dataset. A naive estimate of the imediate impact on revune is of 30\%, comparing to the previous month revenue. The inital analysis suggest that customers who churn tend to have shorter tenures and higher monthly charges than the overall customer population.")

    st.write("In the next tab, we explore these differences in greater detail through an exploratory data analysis aimed at identifying churn patterns in customer demographics characteristics, payment information and subscribed services.")

#impact on revenue.

with tab2:

    ######## First set of plots

    st.write("The following combined violin and box plots help us visualize how the churned customers population differ from non-churned one. Regarding tenure, we find a right-skewed uni-modal distribution for the churned population, with mode close to 3 months. While for the non-churned population, the distribution is bi-modal with peaks close to 5 months and 68 months. This clearly distinctive behaviour signals the company that new clients retention requires imediate attention.")

    st.write("When it comes to monthly charges, we notice that both ditributions are multi-modal. The churned population exhibits higher peaks around \$80 while the highest peak for the non-churned population sits around \$20. This difference highlights that the customers that are prunned to terminate the contract are the ones with higher fees and are not yet fidelized due to low tenure. Thus, it is possible that special offers for the new customers including keeping the prices freezed for a couple of years, directly reduce early tenure churn and offers time to fidelize the customer, positively impacting the revenue in a longer term.")

    st.write("The total charges distributions are the most similar ones when comparing the churned and non-churned populations. Both are uni-modal and right skewed, with peaks somewhere between \$200 and \$600. This aligns with the balance between lower-tenure and higher-fees observed previously for the churned population.")

    # How the charges change with time? What are the highest paid service?
    # ho many services the churned population had??

    option1 = st.selectbox(" ",("Tenure", "Monthly charges", "Total charges"), label_visibility="collapsed")

    mapping1 = {"Tenure":"tenure", "Monthly charges":"MonthlyCharges", "Total charges":"TotalCharges"}
    feat1 = mapping1[option1]

    fig1 = px.violin(processed_data, x="Churn", y=feat1, box=True, category_orders={"Churn":["Yes","No"]}, color="Churn", color_discrete_sequence=["#ef553b","#636efa"])

    y_label={"Tenure":"Tenure (months)", "Monthly charges":"Monthly charges ($)", "Total charges":"Total charges ($)"}

    fig1.update_yaxes(title_text=y_label[option1])
    st.plotly_chart(fig1, width="stretch", key="histograms")

    ######## Second set of plots

    st.write("When it comes to the categorical features, some characteristics stand out. Senior citizens are more pronned to terminating the contract, as well as are customers without partners or dependents. We find 90\% of the customers have phone service but the churn is roughly similar to the proportion of customers that do not have acquired this service. The same holds is true for customers with multiple lines.")

    st.write("The internet services are apparently the deciding factor for terminating the contract. We find Fiber Optics to have the highest association with churn, almost 42\% of Fiber Optics customers have terminated the contract, against 19\% of DSL customers. Regarding online security, online backup, device protection and tech support, the customers that subscribes to these services seem to be less likely to churn. For the customer with streaming movies and streaming TV, there is a less significant association to subscribing to these services and churning, but with a slight tendence to retaining these customers.")

    st.write("The majority of the contracts are month-month, and they also represent the most frequent option of churned customers. On top of that, most clients opt for paperless billing and electronic check payment and both can be liked to higher churn probabilities.")

    option2 = st.selectbox(" ",("Is female or male?", "Is a senior citizen?", "Has a partner?", "Has dependents?", "Has phone service?", "Has multiple lines?", "Has internet service?", "Has online security?", "Has online backup?", "Has device protection?", "Has tech support?", "Has streaming TV?", "Has streaming movies?", "Type of contract?", "Receives paperless billing?", "Pays by which method?"), label_visibility="collapsed")

    mapping2 = {"Is female or male?":"gender", "Is a senior citizen?": "SeniorCitizen", "Has a partner?":"Partner", "Has dependents?":"Dependents", "Has phone service?":"PhoneService", "Has multiple lines?":"MultipleLines", "Has internet service?":"InternetService", "Has online security?" :"OnlineSecurity", "Has online backup?":"OnlineBackup", "Has device protection?":"DeviceProtection", "Has tech support?":"TechSupport", "Has streaming TV?":"StreamingTV", "Has streaming movies?":"StreamingMovies", "Type of contract?":"Contract", "Receives paperless billing?":"PaperlessBilling", "Pays by which method?":"PaymentMethod"}

    feat2 = mapping2[option2]

    agg_eval=processed_data.groupby(by=[feat2,"Churn"]).size().reset_index(name="count")

    total_counts={}
    for cat in agg_eval[feat2].unique():
        total_counts[cat]=agg_eval[agg_eval[feat2]==cat]["count"].sum()

    for i in range(len(agg_eval)):
        agg_eval.loc[i,"percent"]=str(round(100*agg_eval.loc[i,"count"]/total_counts[agg_eval.loc[i,feat2]],1))+"%"

    fig2=px.bar(agg_eval, x=feat2, y="count", color="Churn",text="percent", category_orders={feat2: ["Yes", "No"],"Churn":["Yes","No"]}, barmode="group")
    fig2.update_traces(marker_color="rgba(99, 110, 250, 0.65)", marker_line_color="rgb(99, 110, 250)",marker_line_width=2, selector=dict(name='No'))
    fig2.update_traces(marker_color="rgba(239, 85, 59, 0.65)", marker_line_color="rgb(239, 85, 59)",marker_line_width=2, selector=dict(name='Yes'))
    fig2.update_traces(textposition='outside')
    st.plotly_chart(fig2, width="stretch", key="pie_charts")

    ######## Third plot

    X_num=processed_data[NUMERICAL_FEATURES]
    X_cat=processed_data[BINARY_FEATURES+CATEGORICAL_FEATURES]
    y=processed_data[TARGET]

    le=LabelEncoder()
    y=le.fit_transform(y)

    selector=SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_num,y)
    scores=selector.scores_
    names=selector.feature_names_in_

    #fig4=px.bar(x=scores, y=names,orientation="h")
    #fig4.update_layout(yaxis={'categoryorder':'total ascending'})
    #fig4.update_traces(marker_color="rgba(99, 110, 250, 0.8)", marker_line_color="rgb(99, 110, 250)",marker_line_width=2)
    #fig4.update_layout(margin={'t':10, 'b':10}, height=80,xaxis_title=None, yaxis_title=None)
    #fig4.update_xaxes(visible=False)
    #st.plotly_chart(fig4, width="stretch", key="fig4")


    ############### Teste

    services=processed_data[["PhoneService","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]]
    mapping={"Yes":1, "Fiber optic":1, "DSL":1, "No":0, "No internet service":0}
    for col in services.columns:
        services.loc[:,col]=services.loc[:,col].map(mapping)
    n_of_services=services.sum(axis=1).to_frame()
    n_of_services.columns=["n of services"]

    df_n_services=pd.concat((processed_data[["tenure","MonthlyCharges", "TotalCharges"]],n_of_services,processed_data[["Churn"]]),axis=1)

    services_chart=df_n_services.groupby(by=["Churn","n of services"]).size().reset_index(name="count")

    total_counts={}
    for val in services_chart["n of services"].unique():
        total_counts[val]=services_chart[services_chart["n of services"]==val]["count"].sum()

    for i in range(len(services_chart)):
        services_chart.loc[i,"percent"]=str(round(100*services_chart.loc[i,"count"]/total_counts[services_chart["n of services"][i]],1))+"%"

    fig = px.bar(services_chart, x="n of services", y="count",color="Churn", text="percent", category_orders={"Churn":["Yes","No"]}, barmode="group")
    fig.update_traces(textposition='outside')
    fig.update_traces(marker_color="rgba(99, 110, 250, 0.65)", marker_line_color="rgb(99, 110, 250)",marker_line_width=2, selector=dict(name='No'))
    fig.update_traces(marker_color="rgba(239, 85, 59, 0.65)", marker_line_color="rgb(239, 85, 59)",marker_line_width=2, selector=dict(name='Yes'))
    st.plotly_chart(fig, width="stretch", key="fig")

    ######## Fourth plot


    preprocessor=build_preprocessor(include_scaler=False)

    X_cat=preprocessor.fit_transform(X_cat)

    selector=SelectKBest(score_func=chi2, k='all')
    selector.fit(X_cat,y)
    scores=selector.scores_
    names=[n.split("__")[1] for n in preprocessor.get_feature_names_out()]

    fig5=px.bar(x=scores, y=names, orientation="h", color_discrete_sequence=["#ef553b"])
    fig5.update_layout(yaxis={'categoryorder':'total ascending'})
    fig5.update_traces(marker_color="rgba(99, 110, 250, 0.8)", marker_line_color="rgb(99, 110, 250)",marker_line_width=2)
    fig5.update_layout(margin={'t':10, 'b':10}, height=800,xaxis_title=None, yaxis_title=None)
    fig5.update_xaxes(visible=False)
    st.plotly_chart(fig5, width="stretch", key="fig5")

    ######## Fifth plot

    st.write("Lastly, we can estimate the mutual information between each feature and the target variable. As expected from the previous bar plots, having a month to month contract shows big influence in the decision to churn. This association also holds true for tenure and other internet services as online security and tech support, as well as the the choice of internet itself. The type of contract and the charges are also impactful. It is interesting to note that the customer gender, having a phone service and having multiple lines does not seem to impact the decision to terminate the contract.")

    st.write("In the next tab, we discuss the performance of multiple classification algorithms when trained on this dataset. We will balance between accuracy and maximizing the number of possible churners detected.")

    X_cat=pd.DataFrame(X_cat)
    X_cat.columns=names
    X=pd.concat((X_cat,X_num,n_of_services),axis=1)

    names_all=X.columns
    mask=[]
    for n in names_all:
        if n in X_cat.columns:
            mask.append(True)
        else:
            mask.append(False)

    mi=mutual_info_classif(X, y, discrete_features=mask, random_state=RANDOM_STATE)

    #selector=SelectKBest(score_func=mutual_info_classif, k='all')
    #selector.fit(X,y)
    #scores=selector.scores_

    fig6=px.bar(x=mi, y=names_all, orientation="h", color_discrete_sequence=["#ef553b"])
    fig6.update_layout(yaxis={'categoryorder':'total ascending'})
    fig6.update_traces(marker_color="rgba(99, 110, 250, 0.8)", marker_line_color="rgb(99, 110, 250)",marker_line_width=2)
    fig6.update_layout(margin={'t':10, 'b':10}, height=800,xaxis_title=None, yaxis_title=None)
    fig6.update_xaxes(visible=False)
    st.plotly_chart(fig6, width="stretch", key="fig6")


# FEATURE ENGENEERING
# skeweness could lead to a lot of false negatives.
# Lowering the threshold from 0.5 generally improves Recall, capturing more actual churners — which is the priority in a business context where missing a churner is more costly than a false alarm.

# TAB 3:
# CONFUSION MATRIX
# FEATURE IMPORTANCE
# ROC curve??


with tab4:

    st.write("Now it is your turn to make new predictions. Select the demographic characteristics, subscribed services and payment information and find the churn probability for a customer with these features:")

    with st.form("my_form"):
        col1, col2 = st.columns(2)

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

    st.subheader(f"The probability that this customer will churn is: :primary[{prob}%].")
