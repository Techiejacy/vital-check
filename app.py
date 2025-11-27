import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import re
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer, util
from streamlit_mic_recorder import speech_to_text

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="VitalCheck", page_icon="🩺", layout="wide")

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
FILE_PATHS = {
    "dataset": "data/raw/symptomdatas/dataset.csv",
    "description": "data/raw/symptomdatas/symptom_Description.csv",
    "precaution": "data/raw/symptomdatas/symptom_precaution.csv",
    "who_mortality": "data/processed/who_mortality_sample.csv"
}

# MOVED OUTSIDE cache_resource to ensure updates apply immediately
DISEASE_MAP = {
    '(vertigo) Paroymsal  Positional Vertigo': 'H81', 'AIDS': 'B24', 'Acne': 'L70',
    'Alcoholic hepatitis': 'K70.1', 'Allergy': 'J30', 'Arthritis': 'M13.9',
    'Bronchial Asthma': 'J45', 'Cervical spondylosis': 'M47.9', 'Chicken pox': 'B01',
    'Chronic cholestasis': 'K76.9', 'Common Cold': 'J00', 'Dengue': 'A90',
    'Diabetes ': 'E14', 'Dimorphic hemmorhoids(piles)': 'I84', 'Drug Reaction': 'T88.7',
    'Fungal infection': 'B49', 'GERD': 'K21.9', 'Gastroenteritis': 'A09',
    'Heart attack': 'I21', 'Hepatitis B': 'B18.1', 'Hepatitis C': 'B18.2',
    'Hepatitis D': 'B18.8', 'Hepatitis E': 'B18.8', 'Hypertension ': 'I10',
    'Hyperthyroidism': 'E05.9', 'Hypoglycemia': 'E16.2', 'Hypothyroidism': 'E03.9',
    'Impetigo': 'L01', 'Jaundice': 'R17', 'Malaria': 'B54', 'Migraine': 'G43.9',
    'Osteoarthristis': 'M19.9', 'Paralysis (brain hemorrhage)': 'I61.9',
    'Peptic ulcer diseae': 'K27.9', 'Pneumonia': 'J18.9', 'Psoriasis': 'L40.9',
    'Tuberculosis': 'A16.9', 'Typhoid': 'A01.0', 'Urinary tract infection': 'N39.0',
    'Varicose veins': 'I83.9', 'hepatitis A': 'B15.9'
}

FALLBACK_RISK = {
    # High Risk
    'Heart attack': 1.0, 
    'Paralysis (brain hemorrhage)': 1.0, 
    'AIDS': 0.9, 
    'Tuberculosis': 0.8,
    'Pneumonia': 0.8,
    'Typhoid': 0.8,         # UPDATED: Increased risk (higher than Malaria)
    'Dengue': 0.75,
    'Malaria': 0.7, 
    'Hepatitis B': 0.75, 
    'Hepatitis C': 0.75,
    'Hepatitis D': 0.75,
    'Hepatitis E': 0.7,
    'hepatitis A': 0.65,
    'Alcoholic hepatitis': 0.7,
    
    # Medium Risk
    'Diabetes ': 0.6, 
    'Hypertension ': 0.6,
    'Jaundice': 0.6,
    'Hypoglycemia': 0.5,
    'Bronchial Asthma': 0.5,
    'Hyperthyroidism': 0.4,
    'Hypothyroidism': 0.4,
    'Gastroenteritis': 0.4,
    'Peptic ulcer diseae': 0.4,
    'Chicken pox': 0.4,
    
    # Low/Chronic Risk
    'Arthritis': 0.3,
    'Osteoarthristis': 0.3,
    'Cervical spondylosis': 0.25,
    'Migraine': 0.2,
    'Varicose veins': 0.2,
    'GERD': 0.2,
    'Urinary tract infection': 0.2,
    'Dimorphic hemmorhoids(piles)': 0.2,
    
    # Low/Minor Risk
    'Common Cold': 0.05, 
    'Acne': 0.0, 
    'Allergy': 0.1,
    'Impetigo': 0.1,
    'Fungal infection': 0.1,
    'Drug Reaction': 0.1,
    '(vertigo) Paroymsal  Positional Vertigo': 0.1
}

SYNONYMS = {
    'ache': 'pain', 'hurts': 'pain', 'painful': 'pain', 'sore': 'pain',
    'stomach_ache': 'stomach_pain', 'belly_ache': 'stomach_pain', 'stomach': 'stomach_pain', 'tummy': 'stomach_pain',
    'headache': 'headache', 'chest pain': 'chest_pain', 'chest': 'chest_pain',
    'shivering': 'chills', 'cold': 'chills', 'freezing': 'chills',
    'hot': 'high_fever', 'burning': 'high_fever', 'temp': 'high_fever', 'fever': 'high_fever',
    'sweat': 'sweating', 'perspiring': 'sweating',
    'puke': 'vomiting', 'throw_up': 'vomiting', 'nauseous': 'nausea',
    'urinating': 'polyuria', 'urinate': 'polyuria', 'pee': 'polyuria', 'toilet': 'polyuria', 'frequent urination': 'polyuria',
    'hungry': 'excessive_hunger', 'hunger': 'excessive_hunger', 'starving': 'excessive_hunger', 'eat': 'excessive_hunger',
    'blur': 'blurred_and_distorted_vision', 'blurry': 'blurred_and_distorted_vision', 'vision': 'blurred_and_distorted_vision',
    'dizzy': 'dizziness', 'dizzleness': 'dizziness',
    'breathing': 'breathlessness', 'short_of_breath': 'breathlessness', 'difficulty_breathing': 'breathlessness',
    'rash': 'skin_rash', 'itch': 'itching',
    'mucus': 'phlegm', 'coughing': 'cough', 'catarrh': 'runny_nose', 'sneezing': 'continuous_sneezing'
}

@st.cache_resource
def load_system():
    try:
        for key, path in FILE_PATHS.items():
            if not os.path.exists(path):
                st.error(f"🚨 CRITICAL ERROR: Could not find *{key}* file at: {path}")
                st.stop()

        sym_dataset = pd.read_csv(FILE_PATHS["dataset"]) 
        who_df = pd.read_csv(FILE_PATHS["who_mortality"])
        sym_desc = pd.read_csv(FILE_PATHS["description"])
        sym_prec = pd.read_csv(FILE_PATHS["precaution"])
    except Exception as e:
        st.error(f"🚨 Data Loading Error: {e}")
        st.stop()

    # Clean & Transform Data
    symptom_cols = [col for col in sym_dataset.columns if re.match(r'^Symptom_\d+$', col)]
    df_processed = sym_dataset.copy()
    
    # --- CRITICAL FIX: Clean Disease Column Whitespace ---
    df_processed['Disease'] = df_processed['Disease'].astype(str).str.strip()

    for col in symptom_cols:
        df_processed[col] = df_processed[col].astype(str).str.strip().str.replace(' ', '_').str.lower().replace('none', '')
    
    df_processed['Instance_ID'] = df_processed.index 
    df_long = pd.melt(df_processed, id_vars=['Instance_ID', 'Disease'], value_vars=symptom_cols, value_name='Symptom')
    df_long = df_long[df_long['Symptom'] != '']
    df_long = df_long[df_long['Symptom'] != 'nan']
    df_long['Present'] = 1
    
    X = df_long.pivot_table(index='Instance_ID', columns='Symptom', values='Present', fill_value=0, aggfunc='max')
    Y = df_processed.set_index('Instance_ID')['Disease'].loc[X.index]

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X.values, Y.values)

    # Risk Metadata
    death_cols = [c for c in who_df.columns if 'Deaths' in c]
    who_df['Total_Deaths'] = who_df[death_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    who_agg = who_df.groupby('Cause')['Total_Deaths'].sum().reset_index()
    
    if not who_agg.empty and who_agg['Total_Deaths'].max() > 0:
        who_agg['Risk_Score'] = who_agg['Total_Deaths'] / who_agg['Total_Deaths'].max()
        risk_dict = who_agg.set_index('Cause')['Risk_Score'].to_dict()
    else:
        risk_dict = {}

    def clean_key(k): return str(k).lower().strip().replace(' ', '_').replace('(', '').replace(')', '')
    sym_desc['Disease_Key'] = sym_desc['Disease'].apply(clean_key)
    sym_prec['Disease_Key'] = sym_prec['Disease'].apply(clean_key)

    disease_profiles = {}
    for d in Y.unique():
         d_indices = Y[Y == d].index
         d_rows = X.loc[d_indices]
         symptoms = d_rows.columns[d_rows.sum() > 0].tolist()
         disease_profiles[d] = symptoms

    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
    feature_list = list(X.columns)
    readable_features = [f.replace('_', ' ') for f in feature_list]
    feature_embeddings = nlp_model.encode(readable_features, convert_to_tensor=True)

    return model, feature_list, risk_dict, sym_desc, sym_prec, disease_profiles, nlp_model, feature_embeddings

# Unpack Loaded System
model, features, risk_dict, desc_df, prec_df, disease_profiles, nlp_model, feature_embeddings = load_system()

# ==========================================
# 2. AGENT LOGIC
# ==========================================

def extract_symptoms_hybrid(user_text):
    user_text = user_text.lower()
    found_symptoms = set()
    debug_logs = []

    # Split text by common delimiters
    chunks = re.split(r'[,.]|\band\b|\bbut\b|\balso\b|\bwith\b', user_text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 2]

    for chunk in chunks:
        chunk_matched = False
        
        # 1. SYNONYM MATCH (Using global SYNONYMS)
        for phrase, mapped in SYNONYMS.items():
            pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(pattern, chunk):
                found_symptoms.add(mapped)
                debug_logs.append(f"🔹 Keyword Match: '{phrase}' -> '{mapped}'")
                chunk_matched = True
        
        # 2. EXACT FEATURE MATCH
        for col in features:
            clean = col.replace('_', ' ')
            pattern = r'\b' + re.escape(clean) + r'\b'
            if re.search(pattern, chunk):
                found_symptoms.add(col)
                debug_logs.append(f"🔹 Exact Match: '{col}'")
                chunk_matched = True

        # 3. SEMANTIC AI MATCH
        if not chunk_matched:
            chunk_embedding = nlp_model.encode(chunk, convert_to_tensor=True)
            cosine_scores = util.cos_sim(chunk_embedding, feature_embeddings)[0]
            
            top_results = np.argwhere(cosine_scores.cpu().numpy() > 0.70)
            
            for idx in top_results:
                symptom_idx = idx[0]
                symptom_name = features[symptom_idx]
                found_symptoms.add(symptom_name)
                debug_logs.append(f"🤖 AI Match: '{chunk}' ~ '{symptom_name}' (Score: {cosine_scores[symptom_idx]:.2f})")
            
    return list(found_symptoms), debug_logs

def get_knowledge_response(user_input):
    user_input_lower = user_input.lower()
    known_diseases = {d.lower().strip(): d for d in disease_profiles.keys()}
    known_diseases.update({'sugar': 'Diabetes ', 'pressure': 'Hypertension ', 'flu': 'Common Cold', 'diabetes': 'Diabetes '})

    detected_disease = None
    for key, official_name in known_diseases.items():
        if key in user_input_lower and len(key) > 3: 
            detected_disease = official_name
            break
    
    if detected_disease:
        clean_name = detected_disease.lower().strip().replace(' ', '').replace('(', '').replace(')', '').strip('')
        try: desc = desc_df.loc[desc_df['Disease_Key'] == clean_name, 'Description'].values[0]
        except: desc = "Description unavailable."
        try: 
            p_row = prec_df.loc[prec_df['Disease_Key'] == clean_name].iloc[0, 1:].dropna().tolist()
            remedies = ", ".join([p.strip().capitalize() for p in p_row]) + "."
        except: remedies = "Consult a doctor."
        known_symptoms = disease_profiles.get(detected_disease, [])
        readable_symptoms = [s.replace('_', ' ') for s in known_symptoms]
        symptom_list_str = ", ".join(readable_symptoms[:8]) + ("..." if len(readable_symptoms) > 8 else "")
        return {"type": "disease_info", "name": detected_disease, "desc": desc, "symptoms": symptom_list_str, "remedies": remedies}
    return None

def get_diagnosis(user_input):
    # 1. Custom Intents
    if re.search(r"(who\s+created\s+you|who\s+built\s+you|who\s+made\s+you|who\s+is\s+your\s+creator|developer|jacytech)", user_input, re.IGNORECASE):
        return "Creator", None, None
    if re.search(r"(what\s+are\s+you|what\s+do\s+you\s+do|about\s+yourself|who\s+are\s+you|tell\s+me\s+about\s+you|your\s+purpose)", user_input, re.IGNORECASE):
        return "About", None, None
    if re.search(r"^\s*(hi|hello|hey|greetings|good\s+morning)\b", user_input, re.IGNORECASE): 
        return "Greeting", None, None

    # 2. Knowledge Base Check
    knowledge = get_knowledge_response(user_input)
    if knowledge: return "Knowledge", [knowledge], None

    # 3. Symptom Extraction & Diagnosis
    symptoms, debug_info = extract_symptoms_hybrid(user_input)
    
    with st.sidebar:
        st.subheader("🧠 AI Debugger")
        st.text(f"Input: {user_input}")
        st.text(f"Found: {len(symptoms)}")
        with st.expander("See Logs"):
            for log in debug_info: st.text(log)

    if not symptoms:
        return "Error", [], "No recognizable symptoms found."

    input_data = pd.DataFrame(0, index=[0], columns=features)
    for s in symptoms:
        if s in input_data.columns:
            input_data[s] = 1
            
    probs = model.predict_proba(input_data.values)[0]
    classes = model.classes_
    
    results = []
    best_relevance = 0.0
    
    for i, disease in enumerate(classes):
        prob = probs[i]
        d_clean = disease.strip()
        
        # --- 1. INITIAL RISK LOOKUP (WHO DATA) ---
        icd = DISEASE_MAP.get(d_clean)
        risk_val = risk_dict.get(icd, 0.0)
        
        # --- 2. FALLBACK SYSTEM (If WHO data is missing or near-zero) ---
        # Changed condition from '== 0.0' to '< 0.02' to catch floating point glitches
        if risk_val < 0.02:
            # A. Direct Match
            risk_val = FALLBACK_RISK.get(d_clean, 0.0)
            
            # B. Case-Insensitive Match (The Ultimate Fix)
            if risk_val == 0.0:
                for fb_key, fb_val in FALLBACK_RISK.items():
                    if fb_key.strip().lower() == d_clean.lower():
                        risk_val = fb_val
                        break
        
        # --- 3. HARDCODED SAFETY NETS (For stubborn diseases) ---
        # Explicitly force Malaria if it is still low
        if "malaria" in d_clean.lower() and risk_val < 0.6:
            risk_val = 0.7
        if "typhoid" in d_clean.lower() and risk_val < 0.6:
            risk_val = 0.8  # UPDATED: Increased from 0.6 to 0.8

        profile = disease_profiles.get(disease, [])
        matches = sum(1 for s in symptoms if s in profile)
        
        relevance = matches / len(symptoms) if len(symptoms) > 0 else 0
        profile_len = len(profile)
        coverage = matches / profile_len if profile_len > 0 else 0
        
        if relevance > best_relevance: best_relevance = relevance
        
        score = (relevance * 0.4) + (coverage * 0.3) + (prob * 0.2) + (risk_val * 0.1)
        
        if score > 0.15: # Threshold
            results.append({
                'Disease': disease, 
                'Score': score, 
                'Probability': prob, 
                'Relevance': relevance, 
                'Coverage': coverage,
                'Risk': risk_val
            })
            
    if best_relevance < 0.6: return "Low Match", results, symptoms
    results = sorted(results, key=lambda x: x['Score'], reverse=True)[:3]
    return "Success", results, symptoms

def create_assistant_response(status, results, context):
    """Helper to structure the assistant's data payload."""
    response_content = ""
    chart = None
    msg_data = {"role": "assistant"}

    if status == "Greeting":
        response_content = "👋 Hello! I am an AI health assistant. Please describe your symptoms."
    elif status == "Creator":
        response_content = "I was created by **JacyTech**, a data science and machine learning brand focused on coaching and building intelligent digital solutions.\n\nI'm designed to reflect their mission: *simplifying technology and improving everyday productivity.*"
    elif status == "About":
        response_content = (
            "**“I’m a smart Health Diagnostic Assistant developed by JacyTech, focusing on using machine learning to improve early detection and personal health awareness.**\n\n"
            "I analyze symptoms, highlight possible risk factors, and provide structured insights to help users better understand their health before seeking professional care.\n\n"
            "**My design combines:**\n"
            "* Data science intelligence\n"
            "* Pattern recognition\n"
            "* Risk-aware reasoning\n"
            "* Clear explanations in simple language\n\n"
            "I exist to make health information less confusing, more reliable, and easier to act on. Every part of me reflects JacyTech’s commitment to empowering people with knowledge — because early awareness saves lives.”"
        )
    elif status == "Knowledge":
        res = results[0]
        response_content = f"### 📘 {res['name']}\n\n*Overview:* {res['desc']}\n\n*Typical Symptoms:* {res['symptoms']}\n\n*Remedies:* {res['remedies']}"
    elif status == "Success":
        detected = context
        response_content = f"### 🔍 Analysis Result\n**Symptoms Detected:** {', '.join(detected)}\n\n"
        for i, res in enumerate(results):
             response_content += f"**{i+1}. {res['Disease']}** ({res['Score']*100:.1f}% Confidence)\n"
        # Store Rich Data
        msg_data["results"] = results
        msg_data["detected"] = detected

    elif status == "Low Match":
        response_content = f"🔍 *Detected:* {', '.join(context)}\n\n⚠ *Inconclusive:* Low symptom match. Please see a doctor."
    else:
        response_content = "⚠ I couldn't recognize any symptoms. Try simple keywords like 'fever', 'pain'."
    
    msg_data["content"] = response_content
    return msg_data

# ==========================================
# 3. CHAT UI
# ==========================================

# Sidebar
with st.sidebar:
    st.header("⚙ Controls")
    if st.button("🔄 Reset VitalCheck Memory"):
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    st.header("🎙 Voice Input")
    st.write("Tap to speak:")
    voice_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

st.title("🩺 VitalCheck")
st.caption("Precision Health Assistant")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello. Describe your symptoms (e.g., 'Headache and nausea') or ask about a disease."}]
if "editing_index" not in st.session_state:
    st.session_state.editing_index = None

# Render Chat History
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        
        # --- USER MESSAGE LOGIC (WITH EDITING) ---
        if message["role"] == "user":
            # If this specific message is in "Edit Mode"
            if st.session_state.editing_index == i:
                with st.form(key=f"edit_form_{i}"):
                    new_text = st.text_area("✏️ Edit your symptoms:", value=message["content"])
                    c1, c2 = st.columns([1, 4])
                    if c1.form_submit_button("✅ Update & Rerun"):
                        # 1. Update the User Message
                        st.session_state.messages[i]["content"] = new_text
                        
                        # 2. Re-Run Diagnosis
                        with st.spinner("Updating diagnosis..."):
                            status, results, context = get_diagnosis(new_text)
                            new_response_data = create_assistant_response(status, results, context)
                        
                        # 3. Update the CORRESPONDING Assistant Message (if it exists)
                        # We assume the next message (i+1) is the assistant's reply
                        if i + 1 < len(st.session_state.messages):
                            st.session_state.messages[i+1] = new_response_data
                        else:
                            # If somehow there's no reply yet, append it
                            st.session_state.messages.append(new_response_data)
                        
                        # 4. Exit Edit Mode
                        st.session_state.editing_index = None
                        st.rerun()
                        
                    if c2.form_submit_button("❌ Cancel"):
                        st.session_state.editing_index = None
                        st.rerun()
            else:
                # Normal Display Mode
                col_txt, col_btn = st.columns([15, 1])
                with col_txt:
                    st.markdown(message["content"])
                with col_btn:
                    # Small Edit Button
                    if st.button("✏️", key=f"edit_btn_{i}", help="Edit this message"):
                        st.session_state.editing_index = i
                        st.rerun()

        # --- ASSISTANT MESSAGE LOGIC ---
        else:
            if "results" in message:
                # Render Rich UI (Cards & Charts)
                st.success(f"🔍 *Symptoms Detected:* {', '.join(message['detected'])}")
                
                for idx, res in enumerate(message["results"]):
                    clean_name = res['Disease'].lower().strip().replace(' ', '').replace('(', '').replace(')', '').strip('')
                    try: d_text = desc_df.loc[desc_df['Disease_Key'] == clean_name, 'Description'].values[0]
                    except: d_text = "Description unavailable."
                    try: 
                        p_row = prec_df.loc[prec_df['Disease_Key'] == clean_name].iloc[0, 1:].dropna().tolist()
                        remedies_list = [p.strip().capitalize() for p in p_row]
                    except: remedies_list = ["Consult a doctor."]

                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"### {idx+1}. {res['Disease']}")
                        with c2:
                            if idx == 0:
                                st.markdown(":blue-background[*TOP MATCH*]")
                            else:
                                st.caption(f"Rank #{idx+1}")

                        match_val = res['Score']
                        display_val = min(max(match_val, 0.0), 1.0)
                        st.progress(display_val, text=f"{display_val*100:.1f}% Confidence**")
                        
                        st.caption("DESCRIPTION")
                        st.write(d_text)
                        
                        st.caption("REMEDIES & PRECAUTIONS")
                        rc1, rc2 = st.columns(2)
                        for r_idx, remedy in enumerate(remedies_list):
                            if r_idx % 2 == 0:
                                rc1.markdown(f"✅ {remedy}")
                            else:
                                rc2.markdown(f"✅ {remedy}")
                        
                        with st.expander("View Analysis Data"):
                             sc1, sc2, sc3 = st.columns(3)
                             sc1.metric("Precision", f"{res['Relevance']*100:.1f}%")
                             sc2.metric("Coverage", f"{res['Coverage']*100:.1f}%")
                             sc3.metric("Risk Score", f"{res['Risk']:.2f}")

                # Chart for this specific run
                chart_data = pd.DataFrame(message["results"])
                chart_data['Likelihood'] = chart_data['Score']
                c = alt.Chart(chart_data).mark_bar().encode(
                    x='Likelihood', y=alt.Y('Disease', sort='-x'), color='Risk'
                ).properties(height=200)
                st.altair_chart(c, use_container_width=True)

            else:
                # Standard Text Message
                st.markdown(message["content"])

# --- CHAT INPUT ---
prompt = st.chat_input("Type symptoms (e.g., 'Fever, cough') or ask about JacyTech...")
if voice_text: prompt = voice_text

if prompt:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process Diagnosis
    with st.spinner("Thinking..."):
        status, results, context = get_diagnosis(prompt)
        response_data = create_assistant_response(status, results, context)
    
    # Append Assistant Message
    st.session_state.messages.append(response_data)
    st.rerun()