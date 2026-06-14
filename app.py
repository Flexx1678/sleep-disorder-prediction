import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sleep Disorder Predictor", page_icon="😴", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("sleep_model.pkl")

model = load_model()

DISORDER_CONFIG = {
    "None":        {"emoji": "✅", "color": "#2ecc71", "desc": "Tidak terdeteksi gangguan tidur. Pertahankan pola hidup sehat!"},
    "Insomnia":    {"emoji": "😵", "color": "#e74c3c", "desc": "Terdeteksi kemungkinan Insomnia. Pertimbangkan konsultasi dengan dokter."},
    "Sleep Apnea": {"emoji": "😮‍💨", "color": "#3498db", "desc": "Terdeteksi kemungkinan Sleep Apnea. Segera konsultasikan ke dokter."},
}

# Encoding maps (sesuaikan dengan hasil LabelEncoder di notebook)
GENDER_MAP    = {"Female": 0, "Male": 1}
BMI_MAP       = {"Normal": 0, "Normal Weight": 1, "Obese": 2, "Overweight": 3}
OCCUPATION_MAP = {
    "Accountant": 0, "Doctor": 1, "Engineer": 2, "Lawyer": 3,
    "Manager": 4, "Nurse": 5, "Sales Representative": 6,
    "Salesperson": 7, "Scientist": 8, "Software Engineer": 9, "Teacher": 10
}

st.title("😴 Sleep Disorder Predictor")
st.markdown("Masukkan data gaya hidup untuk memprediksi kemungkinan gangguan tidur.")
st.divider()

st.markdown("#### 👤 Data Pribadi")
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
with col2:
    age = st.number_input("Usia", min_value=18, max_value=80, value=30)
with col3:
    occupation = st.selectbox("Pekerjaan", list(OCCUPATION_MAP.keys()))

st.divider()
st.markdown("#### 😴 Pola Tidur")
col4, col5 = st.columns(2)
with col4:
    sleep_duration = st.slider("Durasi Tidur (jam)", 4.0, 10.0, 7.0, 0.1)
with col5:
    quality_sleep = st.slider("Kualitas Tidur (1-10)", 1, 10, 7)

st.divider()
st.markdown("#### 🏃 Gaya Hidup & Kesehatan")
col6, col7 = st.columns(2)
with col6:
    physical_activity = st.slider("Aktivitas Fisik (menit/hari)", 0, 90, 45)
    stress_level      = st.slider("Tingkat Stres (1-10)", 1, 10, 5)
    daily_steps       = st.number_input("Langkah per Hari", 0, 20000, 7000, 500)
with col7:
    bmi_category = st.selectbox("Kategori BMI", list(BMI_MAP.keys()))
    heart_rate   = st.number_input("Detak Jantung (bpm)", 50, 120, 72)
    bp_systolic  = st.number_input("Tekanan Darah Sistolik", 90, 180, 120)
    bp_diastolic = st.number_input("Tekanan Darah Diastolik", 60, 120, 80)

st.divider()

if st.button("🔍 Prediksi Gangguan Tidur", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([{
        'Gender':                  GENDER_MAP[gender],
        'Age':                     age,
        'Occupation':              OCCUPATION_MAP[occupation],
        'Sleep Duration':          sleep_duration,
        'Quality of Sleep':        quality_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level':            stress_level,
        'BMI Category':            BMI_MAP[bmi_category],
        'Heart Rate':              heart_rate,
        'Daily Steps':             daily_steps,
        'BP Systolic':             bp_systolic,
        'BP Diastolic':            bp_diastolic,
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    classes     = model.classes_
    cfg         = DISORDER_CONFIG[prediction]

    st.markdown(
        f"<div style='background-color:{cfg['color']}22; border-left:5px solid {cfg['color']}; "
        f"padding:16px; border-radius:8px;'>"
        f"<h3 style='color:{cfg['color']}; margin:0'>{cfg['emoji']} {prediction}</h3>"
        f"<p style='margin:4px 0 0 0'>{cfg['desc']}</p></div>",
        unsafe_allow_html=True
    )

    st.markdown("#### Probabilitas per Kelas:")
    prob_df = pd.DataFrame({"Gangguan Tidur": classes, "Probabilitas": [f"{p*100:.1f}%" for p in probability]})
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.warning("⚠️ Hasil ini bukan diagnosis medis. Selalu konsultasikan kondisi kesehatan Anda dengan tenaga medis profesional.")

st.markdown("---")
st.caption("Model: Random Forest Classifier | Dataset: Sleep Health and Lifestyle (374 data)")
