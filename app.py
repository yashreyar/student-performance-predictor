import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Student Performance Predictor")

model = joblib.load("models/pipeline.pkl")

st.title("🎓 Student Performance Prediction System")
st.write("Predict whether a student will pass or fail.")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("Study Hours", 0, 12, 4)
    attendance = st.slider("Attendance (%)", 0, 100, 70)

with col2:
    previous_marks = st.slider("Previous Marks", 0, 100, 60)

if st.button("Predict Result"):

    features = np.array([[study_hours, attendance, previous_marks]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")
