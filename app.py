import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. Load the saved model, scaler, and training column list
#    (these three files must sit in the same folder as this app.py)
# ------------------------------------------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ------------------------------------------------------------------
# 2. Page title
# ------------------------------------------------------------------
st.title("Customer Churn Predictor")
st.write("Enter customer details below to predict churn risk.")

# ------------------------------------------------------------------
# 3. Collect user inputs
# ------------------------------------------------------------------
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
payment_method = st.selectbox(
    "Payment Method",
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
)

# ------------------------------------------------------------------
# 4. On button click: build the input row, scale it, predict
# ------------------------------------------------------------------
if st.button("Predict Churn"):

    # 4a. Start with a single row of zeros, with the SAME columns
    #     (same names, same order) that the model was trained on.
    input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # 4b. Fill in the numeric fields
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly_charges
    input_df["TotalCharges"] = total_charges

    # 4c. Fill in the one-hot encoded contract columns
    #     (only set to 1 if that column exists in training columns)
    if "Contract_One year" in input_df.columns:
        input_df["Contract_One year"] = 1 if contract_type == "One year" else 0
    if "Contract_Two year" in input_df.columns:
        input_df["Contract_Two year"] = 1 if contract_type == "Two year" else 0

    # 4d. Fill in the one-hot encoded Internet Service columns
    #     (dropped category was "DSL", so DSL = all zeros = default)
    if "InternetService_Fiber optic" in input_df.columns:
        input_df["InternetService_Fiber optic"] = 1 if internet_service == "Fiber optic" else 0
    if "InternetService_No" in input_df.columns:
        input_df["InternetService_No"] = 1 if internet_service == "No" else 0

    # 4e. Fill in the one-hot encoded Streaming TV columns
    #     (dropped category was "No", so No = all zeros = default)
    if "StreamingTV_No internet service" in input_df.columns:
        input_df["StreamingTV_No internet service"] = 1 if streaming_tv == "No internet service" else 0
    if "StreamingTV_Yes" in input_df.columns:
        input_df["StreamingTV_Yes"] = 1 if streaming_tv == "Yes" else 0

    # 4f. Fill in the one-hot encoded Payment Method columns
    #     (dropped category was "Bank transfer (automatic)", so that = all zeros = default)
    if "PaymentMethod_Credit card (automatic)" in input_df.columns:
        input_df["PaymentMethod_Credit card (automatic)"] = 1 if payment_method == "Credit card (automatic)" else 0
    if "PaymentMethod_Electronic check" in input_df.columns:
        input_df["PaymentMethod_Electronic check"] = 1 if payment_method == "Electronic check" else 0
    if "PaymentMethod_Mailed check" in input_df.columns:
        input_df["PaymentMethod_Mailed check"] = 1 if payment_method == "Mailed check" else 0

    # 4g. Scale using the SAME scaler fitted during training
    input_scaled = scaler.transform(input_df)

    # 4h. Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # 4i. Show result
    if pred == 1:
        st.error(f"⚠️ High churn risk — {prob*100:.1f}% probability")
    else:
        st.success(f"✅ Low churn risk — {prob*100:.1f}% probability")