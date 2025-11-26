# 🩺 **VitalCheck**
## **Precision Health Diagnostic Powered by AI & WHO Data**


**VitalCheck** is an advanced AI diagnostic assistant that goes beyond traditional symptom checkers.
Instead of relying solely on pattern-matching, it integrates **World Health Organization (WHO) mortality data** to create a *Risk-Aware Intelligence Layer*.

It doesn’t just estimate what users *might* have, it highlights what they **cannot afford to miss**.



## 🌟 **Key Features**

### 🧠 **1. Hybrid Intelligence Engine**

* **Machine Learning:** Random Forest Classifier trained on **4,900+ patient records**.
* **Semantic NLP:** Sentence-Transformers (BERT-based) to understand natural language expressions.

  * e.g., *“my head feels like it’s splitting” → Headache*


### 🛡️ **2. Safety-First Risk Logic**

VitalCheck introduces a unique safety scoring method:

```
Smart Score = (Symptom Match × 70%) 
            + (AI Confidence × 20%) 
            + (WHO Mortality Risk × 10%)
```

High-risk conditions like **Malaria** or **Pneumonia** get prioritised, even when overlapping with harmless illnesses.


### **3. Voice-Enabled Consultation**

Users can speak their symptoms.
Improves accessibility for users with:

* low literacy
* mobility limitations
* visual impairments


### 📊 **4. Transparent Medical Reasoning**

VitalCheck provides clear, interactive visuals:

* **Risk vs. Likelihood charts**
* **Comparison plots for alternative diagnoses**
* **Confidence breakdowns**

No black-box predictions.



## **System Architecture**

```
Input Layer → NLP Extraction → Prediction Layer → Risk Adjustment → Decision Layer → Presentation Layer
```

### 🔍 **1. Input Layer**

User types or speaks symptoms.

### 🧩 **2. NLP Extraction**

Hybrid method:

* Direct keyword detection
* Semantic vector search using BERT

### 🤖 **3. Prediction Layer**

Random Forest Classifier predicts probabilities for **41 diseases**.

### ⚠️ **4. Risk Adjustment Layer**

Pulls WHO mortality statistics and transforms each condition into a **Mortality Risk Score (0–1)**.

### 🎯 **5. Decision Layer**

Applies the Smart Score + relevance threshold to filter low-confidence guesses.

### 📑 **6. Presentation Layer**

Outputs:

* Primary Diagnosis
* Differential Alternatives
* Precautions
* Risk & Likelihood Charts

---

## 🛠️ **Tech Stack**

| Layer            | Technology                               |
| ---------------- | ---------------------------------------- |
| Frontend         | **Streamlit**                            |
| ML Engine        | **Scikit-Learn – Random Forest**         |
| NLP              | **Sentence-Transformers (MiniLM-L6-v2)** |
| Data Processing  | **Pandas, NumPy**                        |
| Visualization    | **Altair**                               |
| Audio Processing | **streamlit-mic-recorder**               |



## **Installation & Setup**

### **Prerequisites**

* Python **3.8+**



### **Clone Repository**

```bash
git clone https://github.com/yourusername/vitalcheck.git
cd vitalcheck
```

---

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

Required libraries:

* streamlit
* scikit-learn
* pandas
* numpy
* sentence-transformers
* streamlit-mic-recorder

---

### **Data Structure**

```
data/
├── raw/
│   └── symptomdatas/
│       ├── dataset.csv
│       └── description.csv
└── processed/
    └── who_mortality_sample.csv
```

---

### **Run Application**

```bash
streamlit run app.py
```

---

## 🔮 **Future Roadmap**

### **Phase 1 — Current**

Diagnostic triage + risk profiling (WHO data integrated).

### **Phase 2 — Q3 2025**

Wearable integration (Fitbit, Apple Watch):

* Real-time heart rate
* Body temperature
* Respiration

### **Phase 3 — Q4 2025**

Telemedicine integration:
Auto hand-off to doctors when **Risk Score > 0.8**.

### **Phase 4 — 2026**

Localized datasets for:

* Tropical regions
* African & Asian disease patterns

---

## ⚠️ **Disclaimer**

VitalCheck is an **AI-based diagnostic aid**, not a medical professional.
Users should consult certified healthcare providers for official medical diagnoses or treatments.
