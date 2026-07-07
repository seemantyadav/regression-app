import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/linear_regression_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

st.title("Insurance Charge Predictor")
st.write("Enter your details below to estimate insurance charges.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
sex_male = st.selectbox("Sex", ["Female", "Male"]) == "Male"
smoker_yes = st.selectbox("Smoker?", ["No", "Yes"]) == "Yes"
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

input_dict = {
    "age": age, "bmi": bmi, "children": children,
    "sex_male": int(sex_male), "smoker_yes": int(smoker_yes),
    "region_northwest": int(region == "northwest"),
    "region_southeast": int(region == "southeast"),
    "region_southwest": int(region == "southwest"),
}
input_df = pd.DataFrame([input_dict])[feature_columns]

if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated insurance charge: ${prediction:,.2f}")