import streamlit as st
import numpy as np
import joblib


st.set_page_config(page_title="Customer Segmentation", page_icon="💳", layout="wide")

scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
svm = joblib.load("SVM.pkl")


st.markdown("<h1 style='text-align: center;'>💳 Customer Segmentation App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>End-to-End ML Pipeline (From Clustering to Classification)</h3>", unsafe_allow_html=True)


st.markdown("#### 📝 Enter the customer details below to predict their segment")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    BALANCE = st.number_input("Balance", min_value=0.0, value=1000.0, step=100.0)
    BALANCE_FREQUENCY = st.number_input("Balance Frequency", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    PURCHASES = st.number_input("Purchases", min_value=0.0, value=500.0, step=50.0)
    ONEOFF_PURCHASES = st.number_input("One-off Purchases", min_value=0.0, value=200.0, step=50.0)
    INSTALLMENTS_PURCHASES = st.number_input("Installments Purchases", min_value=0.0, value=300.0, step=50.0)
    CASH_ADVANCE = st.number_input("Cash Advance", min_value=0.0, value=0.0, step=50.0)

with col2:
    PURCHASES_FREQUENCY = st.number_input("Purchases Frequency", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    ONEOFF_PURCHASES_FREQUENCY = st.number_input("One-off Purchases Frequency", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    PURCHASES_INSTALLMENTS_FREQUENCY = st.number_input("Purchases Installments Frequency", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
    CASH_ADVANCE_FREQUENCY = st.number_input("Cash Advance Frequency", min_value=0.0, max_value=1.5, value=0.1, step=0.01)
    CASH_ADVANCE_TRX = st.number_input("Cash Advance Transactions", min_value=0, value=5, step=1)
    PURCHASES_TRX = st.number_input("Purchases Transactions", min_value=0, value=10, step=1)

with col3:
    CREDIT_LIMIT = st.number_input("Credit Limit", min_value=50.0, value=2000.0, step=100.0)
    PAYMENTS = st.number_input("Payments", min_value=0.0, value=500.0, step=50.0)
    MINIMUM_PAYMENTS = st.number_input("Minimum Payments", min_value=0.0, value=200.0, step=20.0)
    PRC_FULL_PAYMENT = st.number_input("Percent Full Payment", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    TENURE = st.number_input("Tenure (Months)", min_value=6, max_value=12, value=12, step=1)


if st.button("🔮 Predict Customer Segment"):
    
    features = np.array([[BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
                          INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
                          ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
                          CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX,
                          CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS,
                          PRC_FULL_PAYMENT, TENURE]])
    
    
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    
    
    prediction = svm.predict(features_pca)[0]


    segments = {
        0: "💤 Low-spending / Inactive customers — low balances & rare transactions.",
        1: "🛍️ Moderate-spending customers — balanced activity with some bigger buys.",
        2: "💎 High-spending active customers — frequent purchases & high balances."
    }

    st.success(f"🎯 Prediction: {segments[prediction]}")


