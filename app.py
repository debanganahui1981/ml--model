import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('loan_eligibility_model.pkl')

st.title("🏦 Loan Eligibility Prediction App")
st.write("Enter applicant details below to check loan approval status.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
credit_history = st.selectbox("Credit History", ["1.0 (Good)", "0.0 (Poor)"])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount ($ thousands)", min_value=0, value=150)
loan_term = st.number_input("Loan Term (Months)", min_value=0, value=360)

if st.button("Predict Loan Status"):
    # Encoding inputs to match training data
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dep_mapping = {"0": 0, "1": 1, "2": 2, "3+": 3}
    dependents_val = dep_mapping[dependents]
    education_val = 0 if education == "Graduate" else 1
    self_employed_val = 1 if self_employed == "Yes" else 0
    credit_hist_val = 1.0 if "1.0" in credit_history else 0.0
    prop_mapping = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_area_val = prop_mapping[property_area]

    # Log transformations
    applicant_income_log = np.log(applicant_income + 1)
    loan_amount_log = np.log(loan_amount + 1)
    loan_term_log = np.log(loan_term + 1)
    total_income_log = np.log((applicant_income + coapplicant_income) + 1)

    # Feature array matching X column order
    features = np.array([[
        gender_val, married_val, dependents_val, education_val,
        self_employed_val, credit_hist_val, property_area_val,
        applicant_income_log, loan_amount_log, loan_term_log, total_income_log
    ]])

    # Make prediction
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("🎉 Congratulations! Your Loan is Approved.")
    else:
        st.error("❌ Sorry, your Loan application is Rejected.")
