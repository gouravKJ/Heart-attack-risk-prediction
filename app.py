


import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt


model = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
precision, recall = joblib.load("pr_curve.pkl")


st.title("❤️ Heart Attack Risk Prediction")
st.write("Predict heart attack risk using clinical inputs")


name = st.text_input("Enter Your Name")

if name:
    st.success(f"Welcome {name}")

dob = st.date_input("Enter your DOB")
st.write(f"Your DOB: {dob}")

cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
systolic = st.number_input("Systolic BP", 80, 200, 120)
diastolic = st.number_input("Diastolic BP", 50, 150, 80)
heart_rate = st.number_input("Heart Rate", 40, 150, 72)

diabetes = st.selectbox("Diabetes", ["No", "Yes"])
smoking = st.selectbox("Smoking", ["No", "Yes"])
obesity = st.selectbox("Obesity", ["No", "Yes"])

bmi = st.number_input("BMI", 15.0, 45.0, 24.0)


diabetes = 1 if diabetes == "Yes" else 0
smoking = 1 if smoking == "Yes" else 0
obesity = 1 if obesity == "Yes" else 0


if st.button("Predict Heart Attack Risk"):
    input_data = np.array([[
        cholesterol,
        systolic,
        diastolic,
        heart_rate,
        diabetes,
        smoking,
        obesity,
        bmi
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ {name} has HIGH Heart Attack Risk\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ {name} has LOW Heart Attack Risk\n\nProbability: {probability:.2f}")
st.divider()
st.subheader("📈 Model Evaluation – Precision–Recall Curve")

fig, ax = plt.subplots()
ax.plot(recall, precision)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curve")
st.pyplot(fig)


