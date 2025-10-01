import streamlit as st
import pandas as pd
import joblib

st.title("🏦Loan Approval Prediction App")

# Load The Model And Scalar

model = joblib.load("Loan_prd_model.pkl")
scalar = joblib.load("Loan_prd_scaler.pkl")

# user input

age = st.number_input("Enter Age",min_value=20 , max_value=80 , step=1)
income = st.number_input("Enter Your Income",min_value=1000 , max_value=200000, step=10)
assets = st.number_input("Enter Your Assets",min_value=1000 , max_value=1000000 , step=10)
credit_score = st.number_input("Enter Your Credit Score",min_value=1 ,max_value=1000 , step=1)
debt_to_income_ratio = st.number_input("Enter Debt To Income Ratio",min_value=0.1 , max_value=1.0)
existing_loan = st.radio("Existing Loan",[1 , 0])
st.markdown("- ✅ Yes : 1")
st.markdown("- ❌ No : 0")
criminal_record = st.radio("Criminal Record",[1 , 0])

user_input = [[age , income , assets , credit_score , debt_to_income_ratio , existing_loan , criminal_record]]
input_df = pd.DataFrame(user_input , columns=['age', 'income', 'assets', 'credit_score', 'debt_to_income_ratio',
       'existing_loan', 'criminal_record'])

# Scaling the Data

numeric_column =  input_df.select_dtypes(include=['int64','float64']).columns
input_df[numeric_column] = scalar.transform(input_df[numeric_column])

if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.success("Approved")
    else:
        st.success("Not Approved")