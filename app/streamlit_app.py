import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Heart Health AI", page_icon="🫀", layout="wide", initial_sidebar_state="collapsed")

# css
st.markdown("""
    <style>
        /* 1. SAYFA ARKAPLANI */
        .stApp {
            background-color: #3B3736; 
        }
        
        /* 2. HEADER GİZLEME */
        header[data-testid="stHeader"] {
            background-color: transparent;
        }
        .st-emotion-cache-12fmw14 { display: none; }

        /* 3. İÇERİK DÜZENİ */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100%;
        }
        
        /* 4. BAŞLIK */
        h1.big-title {
            font-family: 'Impact', sans-serif;
            color: #d8dcdf;
            font-size: 5vw;
            text-align: center;
            width: 100%;
            margin-top: 0px;
            margin-bottom: 25px;
            line-height: 1.1;
            white-space: nowrap;
            text-transform: uppercase;
            text-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }

        /* 5. FORM KUTUSU */
        [data-testid="stForm"] {
            background-color: #7B382E;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
            height: 100%;
        }

        /* --- submit button  --- */
        
        /* Butonun dış çerçevesi */
        [data-testid="stFormSubmitButton"] button {
            background-color: #aaaaaa !important; 
            color: #dbdcdf !important;              
            border: 2px solid #dbdcdf !important; 
            height: 85px !important;
            border-radius: 15px !important;
            width: 100% !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: all 0.3s ease !important;
        }

        /* Butonun içindeki yazı ve tüm elementler */
        [data-testid="stFormSubmitButton"] button * {
            font-size: 30px !important;
            font-weight: 900 !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            line-height: 1 !important;  
        }
        
        /* Hover */
        [data-testid="stFormSubmitButton"] button:hover {
            background-color: #5c0000 !important; 
            color: #ffcccc !important;           
            border-color: #5c0000 !important;
            transform: scale(1.02) !important;    
            box-shadow: 0 8px 20px rgba(128, 0, 0, 0.4) !important;
        }
            
        /* form metinleri */
        .stMarkdown p, label { 
            color: #dbdcdf !important;  
            font-size: 1.1rem !important; 
            font-weight: 700 !important; 
        }
        
        /* ------ */
            
        div[data-testid="column"]:nth-of-type(2) {
            display: flex;
            flex-direction: column;
        }
            
        div[data-testid="column"]:nth-of-type(2) > div {
            height: 100%;
            width: 100%;
        }

        /* 6. SONUÇ KARTI */
        .result-card {
            height: 100% !important;
            min-height: 500px; 
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            border-radius: 20px;
            border: 1px solid #e0e0e0;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }

        /* ? button */
        div[data-testid="stTooltipContent"] {
            background-color: #75808b !important; 
            color: #ffffff !important;            
            border: 1px solid #75808b;            
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2); 
        }

        div[data-testid="stTooltipContent"] > div, 
        div[data-testid="stTooltipContent"] p {
            color: #ffffff !important;
        }

        [data-testid="stTooltipHoverTarget"] svg {
            fill: #75808b !important;
        }

    </style>
""", unsafe_allow_html=True)

# modeli yükleme
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'heart_disease_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'data_scaler.pkl')

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    return None, None

model, scaler = load_model()

if model is None:
    st.error(f"Error: Model files not found! Looking at: {MODEL_PATH}")
    st.info("Check that the 'models' folder contains your .pkl files.")
else:
    #header
    st.markdown('<h1 class="big-title">HEART DISEASE PREDICTION SYSTEM</h1>', unsafe_allow_html=True)

    #düzen
    left_col, right_col = st.columns([5, 2], gap="large")

    #form
    with left_col:
        with st.form("prediction_form", border=False):        
            st.markdown("""
                <div style="
                    background-color: #75808b; 
                    opacity: 0.8;
                    border-left: 6px solid #75808b; 
                    padding: 15px; 
                    border-radius: 10px; 
                    margin-bottom: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <p style="color: #800000; margin: 0; font-size: 1.1rem; font-weight: 500;">
                        ℹ️ You can see the explanations by clicking the question marks in the upper right corner of the boxes.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                age = st.number_input("Age", 10, 100, 50, help="Biological age of the patient.")
                sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female", help="Biological sex factor.")
                cp = st.selectbox("Chest Pain Type (0-3)", 
                                ["Type 0: No Symptoms", "Type 1: Atypical Angina", "Type 2: Angina", "Type 3: Not Angina"],
                                help="The type of chest pain the patient is complaining of.")
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Blood pressure measured in the hospital (mm Hg).")
                thal = st.selectbox("Thalassemia (1-3)", [1, 2, 3], help="1: Permanent Defect\n2: Normal\n3: Recoverable Defect")

            with col_b:
                chol = st.number_input("Cholesterol (mg/dl)", 100, 500, 200, help="Serum cholesterol level. High values ​​indicate risk.")
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "True" if x == 1 else "False", help="Fasting blood sugar " \
                "above 120 mg/dl indicates a risk of diabetes.")
                restecg = st.selectbox("Resting ECG Result (0-2)", [0, 1, 2], help="0: Normal\n1: ST-T wave abnormality\n2: Hypertrophy (Thickening of the heart " \
                "muscle)")
                thalach = st.number_input("Max Heart Rate", 60, 220, 150, help="The highest heart rate reached during the exercise stress test.")

            with col_c:
                exang = st.selectbox("Exercise Induced Angina?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Do you experience chest pain when " \
                "exercising or running?")
                oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1, help="The amount of depression in the ECG graph after exercise.")
                slope = st.selectbox("ST Slope (0-2)", [0, 1, 2], help="The heart's post-exercise recovery curve.")
                ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3], help="Number of large, unblocked vessels visualized by fluoroscopy.")
                
            st.markdown("---")
            
            #submit butonu
            submit = st.form_submit_button("Analyze Risk", type="primary", use_container_width=True)

            sex_val = 1 if sex == "Male" else 0
            cp_val = int(cp.split(":")[0].split(" ")[1])
            fbs_val = 1 if fbs == "True" else 0
            exang_val = 1 if exang == "Yes" else 0
            

    #sonuç 
    with right_col:
        html_content = ""
        
        if submit:
            if model:
                features = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
                features_scaled = scaler.transform(features)
                probs = model.predict_proba(features_scaled)[0]
                risk_score = probs[1] * 100
                
                if risk_score >= 70:
                    color = "#800000"
                    bg_color = "#f8d7da"
                    border_color = "#800000"
                    status_text = "HIGH RISK DETECTED:"
                    advice = "probability of heart disease."
                elif risk_score >= 30:
                    color = "#CC9E4C"
                    bg_color = "#fffdc2"
                    border_color = "#CC9E4C"
                    status_text = "MODERATE RISK:"
                    advice = "probability of heart disease."
                else:
                    color = "#75808b"
                    bg_color = "#d1e7dd"
                    border_color = "#75808b"
                    status_text = "LOW RISK"
                    advice = "Indicators are within healthy ranges."
                
                html_content = f"""
                    <div class="result-card" style="background-color: 4px solid {bg_color}; border-left: 10px solid {border_color};">
                        <h2 style="color: {color}; margin: 5px 0; font-size: 2.5rem; font-weight: 800;">{status_text}</h2>
                        <h1 style="color: {color}; font-size: 6rem; margin: 10px 0; font-family: 'Arial', sans-serif;">{risk_score:.0f}%</h1>
                        <div style="width: 80%; background-color: #eee; height: 10px; border-radius: 10px; margin: 15px auto;">
                            <div style="width: {risk_score}%; background-color: {color}; height: 10px; border-radius: 10px;"></div>
                        </div>
                        <p style="color: #555; font-size: 1.1rem; margin-top: 15px; padding: 0 10px;">{advice}</p>
                    </div>
                """
        else:
            html_content = """
                <div class="result-card" style="background-color: #d2c0a8; border: 2px dashed #ccc;">
                    <h1 style="font-size: 6rem; margin-bottom: 20px;">🩺</h1>
                    <h3 style="font-size: 2rem; color: #333;">Analysis Awaiting</h3>
                    <p style="font-size: 1.2rem; color: #75808b; margin-top: 10px;">
                        Please fill out the form on the left<br>
                        <b>and click the "ANALYZE RISK" button.</b>
                    </p>
                </div>
            """

        st.markdown(html_content, unsafe_allow_html=True)