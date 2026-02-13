import streamlit as st
import joblib
import numpy as np
import os

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# -------------------------------------------------------
# Load Model (with error handling)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "models/pipeline.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please check deployment.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# -------------------------------------------------------
# App Title
# -------------------------------------------------------
st.title("🎓 Student Performance Prediction System")
st.markdown("Predict whether a student will **PASS or FAIL** based on academic inputs.")

st.markdown("---")

# -------------------------------------------------------
# Input Section
# -------------------------------------------------------
st.subheader("📊 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("Study Hours", 0, 12, 4)
    attendance = st.slider("Attendance (%)", 0, 100, 70)

with col2:
    previous_marks = st.slider("Previous Marks", 0, 100, 60)

# Smart suggestions
if attendance < 50:
    st.warning("⚠ Low attendance may reduce chances of passing.")

if study_hours < 2:
    st.warning("⚠ Very low study hours detected.")

st.markdown("---")

# -------------------------------------------------------
# Prediction Section
# -------------------------------------------------------
if st.button("🔍 Predict Result"):

    features = np.array([[study_hours, attendance, previous_marks]])

    prediction = model.predict(features)

    # If model supports probability
    try:
        proba = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 2)
    except:
        confidence = None

    st.subheader("📢 Prediction Result")

    if prediction[0] == 1:
        if confidence:
            st.success(f"✅ Student is likely to PASS\n\nConfidence: {confidence}%")
        else:
            st.success("✅ Student is likely to PASS")
    else:
        if confidence:
            st.error(f"❌ Student is likely to FAIL\n\nConfidence: {confidence}%")
        else:
            st.error("❌ Student is likely to FAIL")

st.markdown("---")

# -------------------------------------------------------
# Model Information Section
# -------------------------------------------------------
with st.expander("ℹ About This Model"):
    st.write("""
    - This model predicts student performance (Pass/Fail).
    - Features used:
        - Study Hours
        - Attendance
        - Previous Marks
    - Built using Scikit-learn pipeline.
    """)

