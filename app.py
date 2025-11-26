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
# 1. CONFIGURATION & LOADING
# ==========================================
FILE_PATHS = {
    "dataset": "data/raw/symptomdatas/dataset.csv",
    "description": "data/raw/symptomdatas/symptom_Description.csv",
    "precaution": "data/raw/symptomdatas/symptom_precaution.csv",
    "who_mortality": "data/processed/who_mortality_sample.csv"
}

@st.cache_resource
def load_system():
    try:
        for key, path in FILE_PATHS.items():
            if not os.path.exists(path):
                st.error(f"🚨 CRITICAL ERROR: Could not find **{key}** file at: `{path}`")
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

    # Mappings
    disease_map = {
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
    
    fallback_risk = {
        'Heart attack': 1.0, 'Paralysis (brain hemorrhage)': 1.0, 'AIDS': 0.9, 'Tuberculosis': 0.8,
        'Malaria': 0.7, 'Dengue': 0.7, 'Typhoid': 0.6, 'Hepatitis B': 0.7, 'Diabetes ': 0.6, 
        'Pneumonia': 0.7, 'Common Cold': 0.0, 'Acne': 0.0, 'Allergy': 0.1
    }
    
    synonyms = {
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

    return model, feature_list, risk_dict, fallback_risk, disease_map, sym_desc, sym_prec, disease_profiles, nlp_model, feature_embeddings, synonyms

model, features, risk_dict, fallback_risk, mapping, desc_df, prec_df, disease_profiles, nlp_model, feature_embeddings, synonyms = load_system()

# ==========================================
# 2. AGENT LOGIC
# ==========================================

def extract_symptoms_hybrid(user_text):
    user_text = user_text.lower()
    found_symptoms = set()
    debug_logs = []

    chunks = re.split(r'[,.]|\band\b|\bbut\b|\balso\b|\bwith\b', user_text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 2]

    for chunk in chunks:
        chunk_matched = False
        for phrase, mapped in synonyms.items():
            if phrase in chunk:
                found_symptoms.add(mapped)
                debug_logs.append(f"🔹 Keyword: '{phrase}' -> '{mapped}'")
                chunk_matched = True
        
        for col in features:
            clean = col.replace('_', ' ')
            if clean in chunk:
                found_symptoms.add(col)
                debug_logs.append(f"🔹 Exact: '{col}'")
                chunk_matched = True

        if not chunk_matched:
            chunk_embedding = nlp_model.encode(chunk, convert_to_tensor=True)
            cosine_scores = util.cos_sim(chunk_embedding, feature_embeddings)[0]
            top_results = np.argwhere(cosine_scores.cpu().numpy() > 0.55)
            for idx in top_results:
                symptom_idx = idx[0]
                symptom_name = features[symptom_idx]
                found_symptoms.add(symptom_name)
                debug_logs.append(f"🤖 AI Match: '{chunk}' ~ '{symptom_name}'")
            
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
        clean_name = detected_disease.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').strip('_')
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
    greetings = r"^\s*(hi|hello|hey|greetings|good\s+morning)\b"
    if re.match(greetings, user_input, re.IGNORECASE): return "Greeting", None, None

    knowledge = get_knowledge_response(user_input)
    if knowledge: return "Knowledge", [knowledge], None

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
        icd = mapping.get(disease)
        risk_val = risk_dict.get(icd, 0.0)
        if risk_val == 0.0: risk_val = fallback_risk.get(disease, 0.0)
        
        profile = disease_profiles.get(disease, [])
        matches = sum(1 for s in symptoms if s in profile)
        relevance = matches / len(symptoms) if len(symptoms) > 0 else 0
        if relevance > best_relevance: best_relevance = relevance
        
        score = (relevance * 0.7) + (prob * 0.2) + (risk_val * 0.1)
        if score > 0.05:
            results.append({'Disease': disease, 'Score': score, 'Probability': prob, 'Relevance': relevance, 'Risk': risk_val})
            
    if best_relevance < 0.6: return "Low Match", results, symptoms
    results = sorted(results, key=lambda x: x['Score'], reverse=True)[:3]
    return "Success", results, symptoms

# ==========================================
# 3. CHAT UI
# ==========================================

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Reset MediSense Memory"):
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    st.header("🎙️ Voice Input")
    st.write("Tap to speak:")
    voice_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

st.title("🩺 MediSense")
st.caption("Precision Health Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello. Describe your symptoms (e.g., 'Headache and nausea') or ask about a disease."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chart_data" in message:
            st.altair_chart(message["chart_data"], use_container_width=True)

prompt = st.chat_input("Type here...")
if voice_text: prompt = voice_text

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        status, results, context = get_diagnosis(prompt)

    with st.chat_message("assistant"):
        response_content = ""
        chart = None
        
        if status == "Greeting":
            response_content = "👋 Hello! I am ready. Please describe your symptoms."
            
        elif status == "Knowledge":
            res = results[0]
            response_content = f"### 📘 {res['name']}\n\n**Overview:** {res['desc']}\n\n**Typical Symptoms:** {res['symptoms']}\n\n**Remedies:** {res['remedies']}"
            
        elif status == "Success":
            detected = context
            
            # Display detected symptoms nicely at the top
            st.success(f"🔍 **Symptoms Detected:** {', '.join(detected)}")
            st.markdown("---")
            
            # Loop through results to display cards
            for i, res in enumerate(results):
                clean_name = res['Disease'].lower().strip().replace(' ', '_').replace('(', '').replace(')', '').strip('_')
                
                # Get Details
                try: d_text = desc_df.loc[desc_df['Disease_Key'] == clean_name, 'Description'].values[0]
                except: d_text = "Description unavailable."
                try: 
                    p_row = prec_df.loc[prec_df['Disease_Key'] == clean_name].iloc[0, 1:].dropna().tolist()
                    # Create clean list of remedies
                    remedies_list = [p.strip().capitalize() for p in p_row]
                except: remedies_list = ["Consult a doctor."]

                # --- UI CARD DESIGN (Matching the Picture) ---
                with st.container(border=True):
                    # 1. Header Row: Name + Badge
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"### {i+1}. {res['Disease']}")
                    with c2:
                        if i == 0:
                            st.markdown(":blue-background[**TOP MATCH**]")
                        else:
                            st.caption(f"Rank #{i+1}")

                    # 2. Match Percentage Bar
                    match_val = res['Relevance']
                    st.progress(match_val, text=f"**{match_val*100:.1f}% Match**")
                    
                    # 3. Description Section
                    st.caption("DESCRIPTION")
                    st.write(d_text)
                    
                    # 4. Remedies Section (Checklist Style)
                    st.caption("REMEDIES & PRECAUTIONS")
                    rc1, rc2 = st.columns(2)
                    for idx, remedy in enumerate(remedies_list):
                        # Alternate columns for a balanced list look
                        if idx % 2 == 0:
                            rc1.markdown(f"✅ {remedy}")
                        else:
                            rc2.markdown(f"✅ {remedy}")
                    
                    # 5. Technical Stats (Hidden in expander to keep UI clean)
                    with st.expander("View Analysis Data"):
                         sc1, sc2, sc3 = st.columns(3)
                         sc1.metric("AI Confidence", f"{res['Score']*100:.1f}%")
                         sc2.metric("Raw Probability", f"{res['Probability']*100:.1f}%")
                         sc3.metric("Risk Score", f"{res['Risk']:.2f}")

            # Simple Chart at the bottom
           # Simple Chart at the bottom
            st.markdown("### 📊 Comparison Chart")
            chart_data = pd.DataFrame(results)
            chart_data['Likelihood'] = chart_data['Score']
            # Update color encoding to use a Green -> Orange -> Red scale based on Risk
            chart = alt.Chart(chart_data).mark_bar().encode(
                x='Likelihood',
                y=alt.Y('Disease', sort='-x'),
                color=alt.Color('Risk', scale=alt.Scale(range=['green', 'orange', 'red']))
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)
            
        elif status == "Low Match":
            response_content = f"🔍 **Detected:** {', '.join(context)}\n\n⚠️ **Inconclusive:** Low symptom match. Please see a doctor."
        else:
            response_content = "⚠️ One of these symptoms are not in my context. Please try simple keywords like 'fever', 'pain'."

        if status != "Success": st.markdown(response_content)
        
        msg_data = {"role": "assistant", "content": response_content}
        if chart: msg_data["chart_data"] = chart
        st.session_state.messages.append(msg_data)