import streamlit as st
import joblib
import os
from preprocessing import preprocess_input

# --- SETUP ---
st.set_page_config(page_title="Diabetes Risk RF", layout="wide")


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
    return joblib.load(path)


model = load_model()

st.title("🩺 Diabetes Prediction System")
st.markdown("### Random Forest Baseline (Depth: 15)")

with st.form("main_form"):
    # Grouping inputs into 3 columns for better desktop view
    tab1, tab2, tab3 = st.columns(3)

    with tab1:
        st.subheader("Demographics")
        age = st.number_input("Age", 0, 110, 40)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        ethnicity = st.selectbox("Ethnicity", ["White", "Hispanic", "Asian", "Black", "Other"])
        education = st.selectbox("Education", ["Highschool", "Graduate", "Postgraduate", "No formal"])
        income = st.selectbox("Income", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
        employment = st.selectbox("Employment", ["Employed", "Retired", "Student", "Unemployed"])

    with tab2:
        st.subheader("Lifestyle & Vitals")
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        waist_hip = st.number_input("Waist-to-Hip Ratio", 0.5, 1.5, 0.9)
        diet = st.slider("Diet Score", 0.0, 100.0, 50.0)
        alcohol = st.number_input("Alcohol/Week", 0, 50, 2)
        activity = st.number_input("Activity (min/week)", 0, 1000, 150)
        sleep = st.number_input("Sleep (hrs/day)", 0.0, 24.0, 7.0)
        screen = st.number_input("Screen Time (hrs/day)", 0.0, 24.0, 4.0)
        smoking = st.selectbox("Smoking", ["Never", "Current", "Former"])

    with tab3:
        st.subheader("Clinical Metrics")
        sys_bp = st.number_input("Systolic BP", 80, 200, 120)
        dia_bp = st.number_input("Diastolic BP", 40, 130, 80)
        hr = st.number_input("Heart Rate", 40, 200, 72)
        chol = st.number_input("Total Cholesterol", 100, 400, 200)
        hdl = st.number_input("HDL", 20, 100, 50)
        ldl = st.number_input("LDL", 50, 300, 100)
        trig = st.number_input("Triglycerides", 50, 500, 150)

        st.subheader("Medical History")
        fam_diab = st.checkbox("Family History of Diabetes")
        hyp = st.checkbox("History of Hypertension")
        cardio = st.checkbox("History of Cardiovascular Issues")

    submit = st.form_submit_button("Run Prediction Analysis")

if submit:
    # Build dictionary (Keys must match preprocess_input column names)
    data = {
        'age': age, 'alcohol_consumption_per_week': alcohol, 'physical_activity_minutes_per_week': activity,
        'diet_score': diet, 'sleep_hours_per_day': sleep, 'screen_time_hours_per_day': screen, 'bmi': bmi,
        'waist_to_hip_ratio': waist_hip, 'systolic_bp': sys_bp, 'diastolic_bp': dia_bp, 'heart_rate': hr,
        'cholesterol_total': chol, 'hdl_cholesterol': hdl, 'ldl_cholesterol': ldl, 'triglycerides': trig,
        'gender': gender, 'ethnicity': ethnicity, 'education_level': education, 'income_level': income,
        'smoking_status': smoking, 'employment_status': employment,
        'family_history_diabetes': int(fam_diab), 'hypertension_history': int(hyp),
        'cardiovascular_history': int(cardio)
    }

    processed_df = preprocess_input(data)
    prob = model.predict_proba(processed_df)[0][1]

    st.divider()
    if prob > 0.5:
        st.error(f"High Risk: {prob:.2%}")
    else:
        st.success(f"Low Risk: {prob:.2%}")