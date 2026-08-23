# streamlit app
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from predict import predict
from build_pipeline import build_preprocessor
from config import RANDOM_STATE, NUMERICAL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

st.title("Customer Churn Prediction with Machine Learning")
st.write("Identifying customers at high risk of churning in advance enables companies to "
    "develop targeted retention strategies. This project applies supervised machine learning "
    "techniques to estimate customer churn probability for a fictional telecommunications "
    "company using the publicly available IBM Telco Customer Churn dataset.")
st.write("The project workflow includes **data preprocessing, feature engineering, model "
    "training, hyperparameter tuning**, and performance evaluation of multiple classification "
    "models. Given the class imbalance in the dataset, model performance is assessed using "
    "**F1 score** and **ROC-AUC**. The best performing model is integrated into this Streamlit "
    "application, allowing users to generate churn probability estimates for new customers.")
st.write("The source code is publicly available on my [GitHub page]"
    "(https://github.com/CassiaNascimento/Customer_Churn_Prediction).")

tab1, tab2, tab3, tab4 = st.tabs([":open_file_folder: The dataset", 
    ":chart_with_upwards_trend: Exploratory analysis", 
    ":bookmark_tabs: Model performance report", ":keyboard: Make new predictions"])

processed_data=pd.read_csv('../data/processed/cleaned_data.csv')
processed_data['SeniorCitizen']=np.where(processed_data['SeniorCitizen']==0, "No", "Yes")

df_churn=processed_data[processed_data["Churn"]=="Yes"]
df_no_churn=processed_data[processed_data["Churn"]=="No"]

with tab1:

    st.write("Below is the cleaned dataset used in this project, together with a set "
        "of key descriptive metrics.")
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
    col3.metric(label="Churn Rate", value=f"{churn_rate}%", border=True)
    col4.metric(label="Revenue Attrition Rate", value="30.5%",border=True)

    col1, col2 = st.columns(2)
    col1.metric(label="Median Tenure", value=f"{round(median_tenure)} months", border=True)
    col2.metric(label="Median Monthly Charge", value=f"$ {round(median_monthly_charge,2)}", 
        border=True)

    col1, col2 = st.columns(2)
    col1.metric(label="Median Tenure (Churned Customers)", 
        value=f"{round(median_tenure_churn)} months", 
        delta=f"{round(median_tenure_churn-median_tenure)} months", border=True)
    col2.metric(label="Median Monthly Charge (Churned Customers)", 
        value=f"$ {round(median_monthly_charge_churn,2)}", 
        delta=f"${round(median_monthly_charge_churn-median_monthly_charge,2)}", border=True)
    #The churn rate highlights the moderate class imbalance in the dataset.
    st.write("An initial analysis shows that customer churn is responsible for a **30.5% "
        "reduction in monthly revenue**. It also shows that churned customers tend to have "
        "**shorter tenures** and **higher monthly charges** compared to the overall "
        "customer population.")

    st.write("In the next tab, we explore these differences in greater detail through "
        "an exploratory data analysis aimed at identifying churn patterns in customer "
        "demographics characteristics, payment information and subscribed services.")

with tab2:

    ######## First set of plots (numerical features violin and box plots)

    st.write("The following combined violin and box plots help us visualize how the "
        "numerical features behaviour change for the churned customers population:")
    st.write("- **Tenure**: The churned population exhibits a right-skewed uni-modal "
        "distribution, with mode close to **3** months. For the retained customer "
        "population, the distribution is bi-modal with peaks close to **5** months "
        "and **68** months. **Early customer retention** is currently lacking attention.")

    st.write("- **Monthly charges**: Both distributions are multi-modal. The churned "
        "population exhibits higher peaks around **\\$80.00** while the highest peak for "
        "the retained population sits around **\\$20.00**.")
        #, thus the customers that are prone to terminate the contract pay higher fees "
        #"and short tenure.")

    st.write("- **Total charges**: Both distributions are uni-modal and right-skewed, "
        "with peaks between **\\$200.00** and **\\$600.00**. This behaviour aligns with the "
        "balance between shorter-tenure and higher-fees observed previously for the "
        "churned population.")

    option1 = st.selectbox(" ",("Tenure", "Monthly charges", "Total charges"), 
        label_visibility="collapsed")

    mapping1 = {"Tenure":"tenure", "Monthly charges":"MonthlyCharges", 
    "Total charges":"TotalCharges"}
    feat1 = mapping1[option1]

    fig1 = px.violin(processed_data, x="Churn", y=feat1, box=True, 
        category_orders={"Churn":["Yes","No"]}, color="Churn", 
        color_discrete_sequence=["#ef553b","#636efa"])
    y_label={"Tenure":"Tenure (months)", "Monthly charges":"Monthly charges ($)", 
    "Total charges":"Total charges ($)"}
    fig1.update_yaxes(title_text=y_label[option1])
    st.plotly_chart(fig1, width="stretch", key="numerical_features_box_violin_plots")

    ######## Second set of plots (categorical features bar plots)

    st.write("When it comes to the categorical features, several characteristics stand out:")
    st.write("- **Demographics**: Senior citizens are more likely to terminate their contracts, "
        "as are customers without partners and customers without dependents. The subgroup of "
        "customers with neither a partner nor dependents has churn rate of **34.2%**, while the "
        "subgroup of customers with both partners and dependents has churn rate of **14.3%**. "
        "If the customer is a senior citizen subscribed to an internet service and does not have "
        "tech support, the subgroup churn rate reaches **49.4%**, but for senior citizens with "
        "tech support the churn rate is only **19.6%**.") 
    st.write("- **Phone service**: Subscribing to the phone service does not appear to be "
        "associated with churn, and the same holds true for having multiple phone lines.")
    st.write("- **Internet services**: These services appear to be among the main factors "
        "influencing customer churn. For example, the churn rate for customers with Fiber "
        "Optics internet is nearly **42.0\\%**. Regarding online security, online backup, "
        "device protection and tech support, the customers that subscribed to these services "
        "seem to be less likely to churn. In fact, we see that Fiber Optics customers without "
        "Tech Support have a churn rate of **49.4%**, whereas those with Tech Support have a "
        "churn rate of **22.6%**.")
    st.write("- **Contract terms**: Most customers opt for month-to-month contracts, "
        "paperless billing and electronic check as their payment method, all of which "
        "are also associated with higher churn rates. For instance, the churn rate for " 
        "month-to-month contracts is **42.7%**, while for two year contracts it is **2.8%**. "
        "The electronic check payers churn rate is **45.3%**, while for credit card "
        "payers the churn rate is **15.3%**.")

    st.write("For a more detailed discussion, refer to the EDA and Risk Analysis notebooks in "
        "the [project GitHub repository]"
        "(https://github.com/CassiaNascimento/Customer_Churn_Prediction/tree/main/notebooks).")    

    option2 = st.selectbox(" ",("Is female or male?", "Is a senior citizen?", 
        "Has a partner?", "Has dependents?", "Has phone service?", "Has multiple lines?", 
        "Has internet service?", "Has online security?", "Has online backup?", 
        "Has device protection?", "Has tech support?", "Has streaming TV?", 
        "Has streaming movies?", "Type of contract?", "Receives paperless billing?", 
        "Pays by which method?"), label_visibility="collapsed")

    mapping2 = {"Is female or male?":"gender", "Is a senior citizen?": "SeniorCitizen", 
        "Has a partner?":"Partner", "Has dependents?":"Dependents", 
        "Has phone service?":"PhoneService", "Has multiple lines?":"MultipleLines", 
        "Has internet service?":"InternetService", "Has online security?" :"OnlineSecurity", 
        "Has online backup?":"OnlineBackup", "Has device protection?":"DeviceProtection", 
        "Has tech support?":"TechSupport", "Has streaming TV?":"StreamingTV", 
        "Has streaming movies?":"StreamingMovies", "Type of contract?":"Contract", 
        "Receives paperless billing?":"PaperlessBilling", "Pays by which method?":"PaymentMethod"}
    feat2 = mapping2[option2]

    agg_eval=processed_data.groupby(by=[feat2,"Churn"]).size().reset_index(name="count")

    agg_eval["total"] = agg_eval.groupby(by=feat2)["count"].transform("sum")
    agg_eval["percent"] = (agg_eval["count"] / agg_eval["total"]) * 100
    agg_eval["percent_text"] = agg_eval["percent"].round(1).astype(str) + "%"

    fig2=px.bar(agg_eval, x=feat2, y="count", color="Churn",text="percent_text", 
        category_orders={feat2: ["Yes", "No"],"Churn":["Yes","No"]}, barmode="group")
    fig2.update_traces(marker_color="rgba(99, 110, 250, 0.65)", 
        marker_line_color="rgb(99, 110, 250)",marker_line_width=2, selector=dict(name='No'))
    fig2.update_traces(marker_color="rgba(239, 85, 59, 0.65)", 
        marker_line_color="rgb(239, 85, 59)",marker_line_width=2, selector=dict(name='Yes'))
    fig2.update_traces(textposition='outside')
    st.plotly_chart(fig2, width="stretch", key="categorical_features_bar_plots")

    ######## Third plot (number of services bar plot)

    st.write("A closer look at the data reveals that **68.7%** of customers are subscribed "
        "to both internet and phone services. This subgroup accounts for **86.7%** of this "
        "month revenue, with the average revenue per customer being nearly twice that of "
        "customers who only subscribe to internet service and almost four times that of "
        "customers who only subscribe to the phone service. The **churn rate in "
        "this subgroup is 32.8%** and this subgroup is responsible for **93.8% of the "
        "month revenue lost due to churn**.")   

    st.write("These findings motivate an examination of churn rates as a function of number "
        "of services to which customers subscribe. Below graphic shows the churn rate peaks "
        "for customers subscribed to two services and decreases with the increase in number "
        "of services, reaching a minimum of **5.8% among customers subscribed to all 8 "
        "available services**. Customers with only 2, 3 or 4 services sum up to **39.3%** of "
        "total customers and of those, **42.3%** churn.")

    services=processed_data[["PhoneService","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]]
    mapping={"Yes":1, "Fiber optic":1, "DSL":1, "No":0, "No internet service":0}
    for col in services.columns:
        services.loc[:,col]=services.loc[:,col].map(mapping)
    n_of_services=services.sum(axis=1).to_frame()
    n_of_services.columns=["NumberOfServices"]

    df_n_services=pd.concat((processed_data[["tenure","MonthlyCharges", "TotalCharges"]],
        n_of_services,processed_data[["Churn"]]),axis=1)

    services_chart=df_n_services.groupby(by=["Churn","NumberOfServices"]).size().reset_index(name="count")

    total_counts={}
    for val in services_chart["NumberOfServices"].unique():
        total_counts[val]=services_chart[services_chart["NumberOfServices"]==val]["count"].sum()

    for i in range(len(services_chart)):
        services_chart.loc[i,"percent_text"]=str(round(100*services_chart.loc[i,"count"]/total_counts[services_chart["NumberOfServices"][i]],1))+"%"
        services_chart.loc[i,"%"]=100*services_chart.loc[i,"count"]/total_counts[services_chart["NumberOfServices"][i]]

    fig = px.bar(services_chart, x="NumberOfServices", y="%",color="Churn", 
        text="percent_text", category_orders={"Churn":["Yes","No"]}, barmode="group")
    fig.update_traces(textposition='outside')
    fig.update_traces(marker_color="rgba(99, 110, 250, 0.65)", 
        marker_line_color="rgb(99, 110, 250)",marker_line_width=2, selector=dict(name='No'))
    fig.update_traces(marker_color="rgba(239, 85, 59, 0.65)", 
        marker_line_color="rgb(239, 85, 59)",marker_line_width=2, selector=dict(name='Yes'))
    st.plotly_chart(fig, width="stretch", key="churn_number_of_services_plot")

    ######## Fourth plot (customer seniority bar plot)

    st.write("The strong association between churn and short tenure naturally motivates "
        "the development of a customer seniority ranking. One possible approach is to divide "
        "customer tenure into **12 months intervals up to 48 months**. Thus, customers with "
        "tenure shorter than 12 months are assigned seniority level 0, while those with "
        "tenure 48+ months are assigned seniority level 4.") 
    st.write("-**Levels 0 & 1**: This subgroup is formed by 3105 customers and shows a "
        "combined churn rate of **42.1%**.") 
    st.write("-**Levels 2, 3 & 4**: Formed by 3927 customers, the combined churn rate is "
        "**14.3%**.")
    st.write("The findings are summarized in the bar chart below. "
        "It should be noted that customers are not evenly distributed across the "
        "different seniority levels.")
    st.write("For a more detailed discussion, refer to the EDA notebook in the "
        "[project GitHub repository]"
        "(https://github.com/CassiaNascimento/Customer_Churn_Prediction/tree/main/notebooks)")

    processed_data['CustomerLevel'] = pd.cut(processed_data['tenure'],bins=[-np.inf, 11,
     23, 35, 47, np.inf], labels=[0, 1, 2, 3, 4],).astype(int)

    df_grouped=processed_data.groupby(["Churn","CustomerLevel"]).size().reset_index().rename(columns={0:"counts"})
    df_grouped.head()

    total_counts={}
    for val in df_grouped["CustomerLevel"].unique():
        total_counts[val]=df_grouped[df_grouped["CustomerLevel"]==val]["counts"].sum()

    for i in range(len(df_grouped)):
        df_grouped.loc[i,"percent_text"]=str(round(100*df_grouped.loc[i,"counts"]/total_counts[df_grouped["CustomerLevel"][i]],1))+"%"
        df_grouped.loc[i,"%"]=100*df_grouped.loc[i,"counts"]/total_counts[df_grouped["CustomerLevel"][i]]

    # y-axis : percent. This representation causes less awkwardness due to the percents, but does not offer info on the number of customers per seniority level.
    fig7 = px.bar(df_grouped, x="CustomerLevel", y="%",color="Churn", 
        text="percent_text", category_orders={"Churn":["Yes","No"]}, barmode="group")
    fig7.update_traces(textposition='outside')
    fig7.update_traces(marker_color="rgba(99, 110, 250, 0.65)", 
        marker_line_color="rgb(99, 110, 250)",marker_line_width=2, selector=dict(name='No'))
    fig7.update_traces(marker_color="rgba(239, 85, 59, 0.65)", 
        marker_line_color="rgb(239, 85, 59)",marker_line_width=2, selector=dict(name='Yes'))
    st.plotly_chart(fig7, width="stretch", key="churn_customer_seniority_plot")

    # Show which of the plots below will make to the final version

    st.write("Finally, solely as part of the EDA, we evaluate the **Mutual Information (MI)** "
        "score for each feature to measure their non-linear dependency with customer churn:")
    st.write("- Consistent with the previous analyses, the type of **contract** shows "
        "great association with the decision to churn, as does **tenure** and internet "
        "services like **online security** and **tech support**. Other impactful features "
        "are the choice of the **internet** itself and the **number of subscribed services**.")
    st.write("- Customer **gender**, having a **phone service** and having "
        "**multiple lines** do not seem to impact the decision to terminate the contract.")

    X_num=processed_data[NUMERICAL_FEATURES]
    X_cat=processed_data[BINARY_FEATURES+CATEGORICAL_FEATURES]
    names_cat=X_cat.columns
    y=processed_data[TARGET]
    le=LabelEncoder()
    y=le.fit_transform(y)

    encoder = OrdinalEncoder()
    X_cat = encoder.fit_transform(X_cat)
    X_cat=pd.DataFrame(X_cat,columns=names_cat)

    X=pd.concat([X_cat,X_num,n_of_services],axis=1)

    discrete_features=list(X_cat.columns)+list(n_of_services.columns)
    mask = X.columns.isin(discrete_features)
    mi=mutual_info_classif(X, y, discrete_features=mask, random_state=RANDOM_STATE)

    fig6=px.bar(x=mi, y=X.columns, orientation="h", color_discrete_sequence=["#ef553b"])
    fig6.update_layout(yaxis={'categoryorder':'total ascending'})
    fig6.update_traces(marker_color="rgba(99, 110, 250, 0.8)", 
        marker_line_color="rgb(99, 110, 250)",marker_line_width=2)
    fig6.update_layout(margin={'t':10, 'b':10}, height=800,xaxis_title=None, yaxis_title=None)
    fig6.update_xaxes(visible=False)
    st.plotly_chart(fig6, width="stretch", key="mutual_information_plot")

    st.write("In the next tab, we compare the performance of multiple classification "
        "algorithms trained on this dataset, considering the trade-off between"
        "overall predictive accuracy and the ability to correctly identify customers "
        "likely to churn.")

with tab4:

    st.write("Now it is your turn to make new predictions. Select the demographic "
        "characteristics, subscribed services and payment information and find the churn "
        "probability for a customer with these features:")

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
        option15 = st.selectbox("What is the type of contract?",("Month-to-month", "One year", 
            "Two year"))
        option16 = st.selectbox("Receives paperless billing?",("Yes","No"))
        option17 = st.selectbox("Pays by which method?",("Bank transfer (automatic)", 
            "Credit card (automatic)", "Electronic check", "Mailed check"))
        option5 = st.slider("What is the contract duration?",0,75,25)
        option18 = st.slider("What are the Monthly Charges?",0,150,80)
        option19 = st.slider("What are the Total Charges?",0,8500,2000)

        mapping={"Yes":1,"No":0}

        submitted=st.form_submit_button("Predict")

        if submitted:
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
