
import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import joblib

# ================== Load Files =====================
# Load symptom severity
severity_df = pd.read_csv("Symptom-severity.csv")
severity_df.columns = severity_df.columns.str.strip().str.replace('\ufeff', '')
symptom_severity = dict(zip(severity_df['Symptom'].str.lower().str.strip(), severity_df['weight']))

# Load precautions
precaution_df = pd.read_csv("symptom_precaution.csv").fillna("")
precaution_df.columns = precaution_df.columns.str.strip().str.replace('\ufeff', '')
precaution_dict = {row["Disease"]: [row["Precaution_1"], row["Precaution_2"], row["Precaution_3"], row["Precaution_4"]] for _, row in precaution_df.iterrows()}

# Load X feature names
df = pd.read_csv("dataset.csv").fillna("")
all_symptoms = sorted(set(symptom.lower().strip() for col in df.columns[1:] for symptom in df[col].unique() if symptom))
symptom_index = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

# Load models
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load columns
X = pd.DataFrame(columns=all_symptoms)

# 🩺 Set page config
st.set_page_config(page_title="Disease Prediction App", page_icon="🩺")

st.title("🩺 Disease Prediction from Symptoms")

# ✅ SYMPTOM LIST - Add your list of symptoms here
SYMPTOM_LIST = sorted([
    'itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 'shivering',
    'chills', 'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting',
    'vomiting', 'burning micturition', 'spotting urination', 'fatigue', 'weight gain',
    'anxiety', 'cold hands and feets', 'mood swings', 'weight loss', 'restlessness',
    'lethargy', 'patches in throat', 'irregular sugar level', 'cough', 'high fever',
    'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion'
    # 👉🏻 Add more symptoms based on your dataset!
])

# 👉🏻 Multi-select symptom input
selected_symptoms = st.multiselect(
    "Select your Symptoms:",
    SYMPTOM_LIST
)

def predict_multiple_diseases(selected_symptoms):
    input_list = [0] * len(symptom_index)
    
    for s in selected_symptoms:
        s_clean = s.lower().strip().replace(" ", "_")
        if s_clean in symptom_index:
            input_list[symptom_index[s_clean]] = symptom_severity.get(s_clean.replace("_", " "), 0)
    
    input_df = pd.DataFrame([input_list], columns=X.columns)
    
    pred_rf = encoder.classes_[rf_model.predict(input_df)[0]]
    pred_nb = encoder.classes_[nb_model.predict(input_df)[0]]
    pred_svm = encoder.classes_[svm_model.predict(input_df)[0]]
    
    all_preds = [pred_rf, pred_nb, pred_svm]
    prediction_counts = Counter(all_preds)
    top_predictions = prediction_counts.most_common(3)
    
    result = []
    for disease, count in top_predictions:
        precautions = precaution_dict.get(disease, ["Not available"] * 4)
        result.append({
            "Disease": disease,
            "Votes": count,
            "Precautions": precautions
        })
    
    return result

if st.button("Predict Disease"):
    if selected_symptoms:
        results = predict_multiple_diseases(selected_symptoms)
        for idx, res in enumerate(results):
            st.subheader(f"Prediction #{idx+1}: {res['Disease']}")
            st.write(f"Votes: {res['Votes']} / 3")
            st.markdown("**Precautions:**")
            for p in res['Precautions']:
                st.write(f"- {p}")
    else:
        st.warning("Please select at least one symptom!")
