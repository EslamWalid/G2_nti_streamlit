import streamlit as st
import pandas as pd
import joblib


model = joblib.load('logistic_regression_model.pkl')
label_encoder = joblib.load('label_encoders.pkl')
features =['job', 'marital', 'education', 'contact', 'month', 'poutcome'] 


st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Marketing Prediction")
st.write("Enter the customer information to predict the campaign outcome.")


col1, col2 = st.columns(2)

with col1:
    marital = st.selectbox(
        "Marital Status",
        ["married", "divorced", "single"]
    )

    job = st.selectbox(
        "Job",
        [
            "admin.",
            "unknown",
            "unemployed",
            "management",
            "housemaid",
            "entrepreneur",
            "student",
            "blue-collar",
            "self-employed",
            "retired",
            "technician",
            "services"
        ]
    )

    education = st.selectbox(
        "Education",
        ["unknown", "secondary", "primary", "tertiary"]
    )



with col2:
    contact = st.selectbox(
        "Contact Type",
        ["unknown", "telephone", "cellular"]
    )

    month = st.selectbox(
        "Last Contact Month",
        [
            "jan", "feb", "mar", "apr",
            "may", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec"
        ]
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "other", "failure", "success"]
    )


data = pd.DataFrame({
    'job': [job],
    'marital': [marital],
    'education': [education],
    'contact': [contact],
    'month': [month],
    'poutcome': [poutcome]
})


if st.button("Predict"):

    for feature in features:
        data[feature] = label_encoder[feature].transform(data[feature])

    prediction = model.predict(data)

    if prediction[0] == 1:
        prediction_text = "The customer is likely to subscribe to the term deposit. YES"
    else:
        prediction_text = "The customer is unlikely to subscribe to the term deposit. NO"

    st.write(prediction_text)
