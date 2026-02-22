import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import time
import base64
from io import BytesIO

# --- 1. PAGE CONFIGURATION & ADVANCED CSS ---
st.set_page_config(
    page_title="NeuroDerma AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown(
        """
        <style>
        h1, h2, h3 { color: #0072B5; font-family: 'Helvetica Neue', sans-serif; }
        .css-1r6slb0 { background-color: #FFFFFF; border-radius: 15px; padding: 20px; box-shadow: 0px 4px 12px rgba(0,0,0,0.05); border: 1px solid #F0F2F6; }
        section[data-testid="stSidebar"] { box-shadow: 2px 0px 10px rgba(0,0,0,0.05); background-color: #FAFBFC; }
        .stButton>button { border-radius: 25px; font-weight: 600; border: none; box-shadow: 0px 2px 5px rgba(0,0,114,0.2); transition: all 0.3s ease; }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0px 4px 8px rgba(0,0,114,0.3); }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #FFFFFF; border-radius: 10px 10px 0px 0px; box-shadow: 0px -2px 5px rgba(0,0,0,0.02); font-weight: 600; }
        .stTabs [aria-selected="true"] { color: #0072B5; background-color: #F0F8FF; }
        [data-testid="stMetricValue"] { color: #0072B5; font-weight: 700; }
        
        /* SCANNER ANIMATION CSS */
        .scanner-container { position: relative; width: 100%; border-radius: 15px; overflow: hidden; box-shadow: 0 8px 20px rgba(0,0,0,0.1); border: 2px solid #E6E9EF; }
        .scanner-container img { width: 100%; display: block; }
        .scanner-line { position: absolute; top: 0; left: 0; width: 100%; height: 4px; background-color: #00C6CF; box-shadow: 0 0 15px #00C6CF, 0 0 30px #00C6CF; animation: scan 2s ease-in-out infinite; }
        @keyframes scan { 0% { top: 0%; opacity: 0; } 10% { opacity: 1; } 50% { top: calc(100% - 4px); } 90% { opacity: 1; } 100% { top: 0%; opacity: 0; } }
        </style>
        """,
        unsafe_allow_html=True
    )
local_css()

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if 'history' not in st.session_state:
    st.session_state.history = []
if 'scan_completed' not in st.session_state:
    st.session_state.scan_completed = False
if 'last_filename' not in st.session_state:
    st.session_state.last_filename = ""
if 'view_index' not in st.session_state:
    st.session_state.view_index = -1

DISEASE_INFO = {
    "BA- cellulitis": {
        "name": "Cellulitis", 
        "type": "Bacterial Infection", 
        "description": "A potentially serious bacterial skin infection that causes redness, swelling, and pain.\n\n**🔍 Common Symptoms:** Red area of skin that tends to expand, swelling, tenderness, warmth, and fever.\n\n**🤝 Contagion Risk:** Not contagious. It is an internal infection of the deeper skin layers.", 
        "action": "Seek medical attention immediately. Requires prescription antibiotics."
    },
    "BA-impetigo": {
        "name": "Impetigo", 
        "type": "Bacterial Infection", 
        "description": "A highly contagious skin infection mainly affecting children, appearing as red sores.\n\n**🔍 Common Symptoms:** Red sores that quickly rupture, ooze for a few days, and then form a yellowish-brown crust.\n\n**🤝 Contagion Risk:** Highly contagious through direct contact or sharing towels/clothing.", 
        "action": "Consult a doctor for prescription antibiotic ointment or oral antibiotics."
    },
    "FU-athlete-foot": {
        "name": "Athlete's Foot", 
        "type": "Fungal Infection", 
        "description": "A fungal infection usually beginning between the toes due to sweaty feet confined in tight shoes.\n\n**🔍 Common Symptoms:** Scaly, peeling, or cracked skin between the toes with intense itchiness.\n\n**🤝 Contagion Risk:** Contagious. Spreads easily via contaminated floors, towels, or clothing.", 
        "action": "Use over-the-counter antifungal creams. Keep feet dry and change socks regularly."
    },
    "FU-nail-fungus": {
        "name": "Nail Fungus", 
        "type": "Fungal Infection", 
        "description": "A common condition beginning as a white/yellow spot under the nail tip, causing discoloration and thickening.\n\n**🔍 Common Symptoms:** Thickened, brittle, crumbly, or ragged nails with distorted shape and dark color.\n\n**🤝 Contagion Risk:** Mildly contagious. Can spread from toe to toe, or occasionally person to person.", 
        "action": "Consult a dermatologist/podiatrist for prescription oral antifungal pills."
    },
    "FU-ringworm": {
        "name": "Ringworm", 
        "type": "Fungal Infection", 
        "description": "A common fungal skin infection causing a ring-shaped, red, itchy rash.\n\n**🔍 Common Symptoms:** A ring-shaped red, scaly patch with a clear or scaly center that may expand outward.\n\n**🤝 Contagion Risk:** Highly contagious via skin-to-skin contact or touching infected objects/pets.", 
        "action": "Apply over-the-counter antifungal cream. See doctor if not cleared in 2-4 weeks."
    },
    "PA-cutaneous-larva-migrans": {
        "name": "Cutaneous Larva Migrans", 
        "type": "Parasitic Infection", 
        "description": "Caused by hookworm larvae, appearing as intensely itchy, twisting red lines.\n\n**🔍 Common Symptoms:** Raised, snake-like, twisting red tracks on the skin that migrate slightly each day.\n\n**🤝 Contagion Risk:** Not contagious from human to human.", 
        "action": "Consult healthcare provider for anti-parasitic medications."
    },
    "VI-chickenpox": {
        "name": "Chickenpox", 
        "type": "Viral Infection", 
        "description": "Highly contagious viral infection causing an itchy, blister-like rash.\n\n**🔍 Common Symptoms:** An itchy rash with fluid-filled blisters that turn into scabs, often preceded by fever and tiredness.\n\n**🤝 Contagion Risk:** Highly contagious to those who haven't had the disease or vaccine.", 
        "action": "Rest, stay hydrated, use calamine lotion. Avoid scratching."
    },
    "VI-shingles": {
        "name": "Shingles", 
        "type": "Viral Infection", 
        "description": "Painful rash caused by reactivation of the chickenpox virus.\n\n**🔍 Common Symptoms:** Pain, burning, or tingling followed by a red rash and fluid-filled blisters that crust over.\n\n**🤝 Contagion Risk:** The rash itself cannot pass Shingles, but can transmit the virus to cause Chickenpox in non-immune individuals.", 
        "action": "See a doctor immediately for antiviral medications (best within 72 hours)."
    }
}
CLASS_NAMES = list(DISEASE_INFO.keys())

@st.cache_resource
def load_tflite_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))   # folder where app.py is
    model_path = os.path.join(base_dir, "model.tflite")

    if not os.path.exists(model_path):
        # Shows exactly where it tried to look (super helpful on Streamlit Cloud)
        raise FileNotFoundError(f"model.tflite not found at: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Model Error: {e}")
    st.stop()

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def render_dashboard(data):
    tab1, tab2, tab3 = st.tabs(["✨ Clinical Analysis", "📊 Model Insights", "🏥 Action Plan & Specialist"])

    with tab1:
        col_img, col_stats = st.columns([1, 1.5], gap="large")
        with col_img:
            st.image(data['image'], use_column_width=True, caption=f"Analyzed at {data['time']}")
        
        with col_stats:
            if data['conf'] > 95: st.success("✅ **High Confidence Match Detected**")
            elif data['conf'] > 80: st.info("ℹ️ **Moderate Confidence - Clinical Correlation Advised**")
            else: st.warning("⚠️ **Low Confidence - Specialist Consultation Highly Recommended**")

            m1, m2 = st.columns(2)
            m1.metric("Primary Detection", data['name'])
            m2.metric("Confidence Score", f"{data['conf']:.2f}%")
            st.progress(int(data['conf']))
            st.markdown(f"**Pathology Type:** {data['type']}")

            report_text = f"NeuroDerma AI Report\nTime: {data['time']}\nDetection: {data['name']}\nConfidence: {data['conf']:.2f}%\n\nDescription:\n{data['description']}\n\nAction:\n{data['action']}"
            st.download_button(label="📥 Download Case Report", data=report_text, file_name=f"Report_{data['time'].replace(':','')}.txt", key=f"dl_{data['time']}")

    with tab2:
        st.markdown("### 🧠 Underlying Probability Distribution")
        st.write("View the model's top 3 competing predictions to understand uncertainty boundaries.")
        st.bar_chart(data['chart_data'].set_index("Condition"))

    with tab3:
        c1, c2 = st.columns(2, gap="large")
        with c1:
             st.markdown("### 📚 Medical Guidance")
             with st.container():
                st.markdown("**Description:**")
                st.write(data['description'])
                st.divider()
                st.markdown("**⚕️ Recommended Action:**")
                st.info(data['action'])
        with c2:
            st.markdown("### 📍 Specialist Finder")
            

            maps_html = """
            <div style="font-family: 'Helvetica Neue', sans-serif;">
                <p style="color: #262730; font-size: 14px; margin-bottom: 12px;">Locate certified dermatologists nearby for professional diagnosis.</p>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <input type="text" id="map_loc" placeholder="Enter City or Zip Code..." 
                           style="flex: 1; padding: 12px 15px; border: 1px solid #E6E9EF; border-radius: 25px; outline: none; font-size: 14px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);">
                    <button onclick="var loc = document.getElementById('map_loc').value; if(loc) { window.open('https://www.google.com/maps/search/dermatologist+near+' + encodeURIComponent(loc), '_blank'); } else { alert('Please enter a location first.'); }" 
                            style="background-color: #0072B5; color: white; border: none; padding: 12px 20px; border-radius: 25px; cursor: pointer; font-weight: 600; font-size: 14px; transition: 0.3s; box-shadow: 0px 2px 5px rgba(0,0,114,0.2);">
                        🗺️ Launch Google Maps
                    </button>
                </div>
            </div>
            """
            components.html(maps_html, height=120)


with st.sidebar:
    st.title("NeuroDerma AI")
    st.divider()
    
    st.markdown("### 🕒 Session History")
    if not st.session_state.history:
        st.info("No analyses run yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            real_index = len(st.session_state.history) - 1 - i
            with st.expander(f"[{item['time']}] {item['name']}"):
                st.write(f"Confidence: {item['conf']:.2f}%")
                if st.button("Load Analysis", key=f"load_{real_index}"):
                    st.session_state.view_index = real_index
                    st.session_state.scan_completed = True

    st.divider()
    st.warning("⚠️ **Disclaimer:** Deep Learning Project, check with a Doctor for actual diagnosis.")


st.markdown("# 🩺 NeuroDerma AI Analysis")

uploaded_file = st.file_uploader("Drop image here, or browse", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_filename:
        st.session_state.last_filename = uploaded_file.name
        st.session_state.scan_completed = False 
        st.session_state.view_index = -1        

if st.session_state.view_index != -1:
    st.info("💡 **Viewing Past Analysis from History**")
    render_dashboard(st.session_state.history[st.session_state.view_index])

elif uploaded_file is None:
    col1, col2 = st.columns([1.5, 1])
    with col2:
        st.markdown("### 📋 Best Practices")
        st.info("💡 **Bright, even lighting:** Avoid harsh shadows.")
        st.info("📸 **Sharp focus:** Ensure the lesion is clearly visible.")
        st.info("📏 **About 10cm distance:** Don't get too close or too far.")

else:
    image = Image.open(uploaded_file).convert("RGB")
    
    if not st.session_state.scan_completed:
        animation_container = st.empty()
        with animation_container.container():
            scan_col1, scan_col2 = st.columns([1, 1], gap="large")
            with scan_col1:
                img_b64 = image_to_base64(image)
                st.markdown(f'<div class="scanner-container"><img src="data:image/jpeg;base64,{img_b64}"><div class="scanner-line"></div></div>', unsafe_allow_html=True)
                
            with scan_col2:
                st.markdown("<br><br><br> 🧠 Analyzing Image...", unsafe_allow_html=True)
                step1, step2, step3 = st.empty(), st.empty(), st.empty()
                
                step1.markdown("⏳ Preprocessing image quality...")
                time.sleep(1.0)
                step1.markdown("✅ **Preprocessing image quality**")
                
                step2.markdown("⏳ Extracting dermatological features...")
                time.sleep(1.0)
                step2.markdown("✅ **Extracting dermatological features**")
                
                step3.markdown("⏳ Running deep learning classification...")
                
                input_data = preprocess_image(image)
                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]["index"])[0]
                
                top_index = int(np.argmax(output_data))
                top_class = CLASS_NAMES[top_index]
                confidence = float(np.max(output_data)) * 100
                disease_data = DISEASE_INFO[top_class]
                
                top_3_indices = output_data.argsort()[-3:][::-1]
                chart_data = pd.DataFrame({
                    "Condition": [DISEASE_INFO[CLASS_NAMES[i]]['name'] for i in top_3_indices],
                    "Probability (%)": output_data[top_3_indices] * 100
                })
                
                time.sleep(0.5)
                step3.markdown("✅ **Running deep learning classification**")
                time.sleep(0.5)

        timestamp = time.strftime("%H:%M:%S")
        analysis_data = {
            'image': image, 'name': disease_data['name'], 'conf': confidence,
            'type': disease_data['type'], 'description': disease_data['description'],
            'action': disease_data['action'], 'chart_data': chart_data, 'time': timestamp
        }
        

        st.session_state.history.append(analysis_data)
        st.session_state.scan_completed = True
        
        st.experimental_rerun()
        
    else:
        render_dashboard(st.session_state.history[-1])

