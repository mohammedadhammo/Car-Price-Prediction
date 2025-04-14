import streamlit as st
import joblib
import numpy as np

import xgboost as xgb

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("xgb_model.json")  # لو الملف في نفس الفولدر
    return model

model = load_model()


# إعداد الصفحة
st.set_page_config(page_title="Car Price Predictor", layout="centered")

# --- CSS لتغيير خلفية الصفحة بالكامل + تنسيق داخلي ---
st.markdown("""
    <style>
    /* الخلفية العامة */
    body, .stApp {
        background-color: #e6f0f8 !important;
    }

    /* الديف الرئيسي */
    .main-container {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
        animation: fadeZoom 1s ease-in-out;
        max-width: 900px;
        margin: auto;
    }

    /* تخصيص التايتل */
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        animation: fadeInDown 1s ease-in-out;
        color: #000000;  /* اللون الأسود */
    }
    /* تخصيص الصب تايتل */
    .sub-title {
        text-align: center;
        margin-bottom: 30px;
        animation: fadeInUp 1s ease-in-out;
        color: #000000;  /* اللون الأسود */
    }
    /* تخصيص عنوان الحقول */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #000000;  /* اللون الأسود */
        font-weight: bold;
    }

    /* تخصيص الـ selectbox */
    .stSelectbox select {
        color: #ffffff !important;  /* تغيير النص داخل الـ selectbox إلى اللون الأبيض */
        background-color: #2d3e50 !important;  /* تغيير خلفية الـ selectbox إلى لون غامق */
        border-radius: 5px;
        padding: 5px;
    }

    /* تغيير النص داخل الـ selectbox (العناصر) */
    .stSelectbox select option {
        color: #000000 !important;  /* تغيير اللون داخل العناصر المنسدلة إلى الأسود */
    }

    /* النص داخل خانات الإدخال أبيض */
    input, select, textarea {
        color: #ffffff !important;
    }

    /* تغيير النص داخل خانات الإدخال إلى اللون الأبيض */
    .stTextInput input, .stNumberInput input, .stSlider input {
        color: #ffffff !important;  /* تغيير النص داخل خانات الإدخال إلى اللون الأبيض */
    }

    /* تخصيص الـ slider */
    .stSlider input {
        color: #ffffff !important; /* تغيير اللون الأبيض للنص في الـ slider */
    }

    /* تغيير النص داخل النتائج إلى اللون الأسود */
    .result {
        font-size: 22px;
        font-weight: bold;
        background-color: #d6eaf8;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        animation: fadeIn 1s ease-in-out;
        color: #000000;  /* اللون الأسود */
    }

    /* باقي التنسيقات */
    .logo {
        display: block;
        margin: 0 auto 20px auto;
        width: 120px;
        border-radius: 15px;
        animation: fadeIn 1.2s ease-in-out;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white !important;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1c5985;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeZoom {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)


# فتح الديف الكبير
# st.markdown('<div class="main-container">', unsafe_allow_html=True)

# شعار
st.markdown('''
    <img class="logo" src="https://i.postimg.cc/gJ0mr9gZ/8161894.jpg" alt="Logo" style="width: 320px;  height: 170px; border-radius: 15px; display: block; margin: 0 auto 20px auto;"/>
''', unsafe_allow_html=True)


# عناوين
st.markdown('<div class="main-title">Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter car details to get the estimated price (in USD)</div>', unsafe_allow_html=True)

# مدخلات المستخدم
col1, col2 = st.columns(2)

with col1:
    model_number = st.number_input("Model Code", min_value=0, max_value=1000, value=100)
    mpg = st.number_input("Fuel Efficiency (MPG)", min_value=0.0, max_value=150.0, value=50.0, step=0.5)
    engine_size = st.number_input("Engine Size (Liters)", min_value=0.5, max_value=6.0, value=1.6, step=0.1)
    tax = st.number_input("Tax ($)", min_value=0, max_value=1000, value=150)

with col2:
    year = st.slider("Year", 1990, 2025, 2020)
    mileage = st.number_input("Mileage (KM)", min_value=0, max_value=500000, value=50000, step=1000)
    transmission_type = st.selectbox("Transmission Type", ["Automatic", "Manual", "Semi-Automatic"])

# تحويل النقل إلى أرقام
trans_manual = 1 if transmission_type == "Manual" else 0

# تشكيل المدخلات
input_data = np.array([[model_number, mpg, engine_size, year, mileage, trans_manual, tax]])

# زر التوقع
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.markdown(
        f'<div class="result">Estimated Price: ${prediction[0]:,.2f}</div>',
        unsafe_allow_html=True
    )

# قفل الديف
st.markdown('</div>', unsafe_allow_html=True)
