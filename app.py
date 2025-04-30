import streamlit as st
import pandas as pd
import pickle
from collections import Counter
import numpy as np

# ================== Load Models and Data =====================

@st.cache_resource(show_spinner=False)
def load_models():
    with open("svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("nb_model.pkl", "rb") as f:
        nb_model = pickle.load(f)
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("symptom_index.pkl", "rb") as f:
        symptom_index = pickle.load(f)
    with open("precaution_dict.pkl", "rb") as f:
        precaution_dict = pickle.load(f)
    return svm_model, nb_model, rf_model, encoder, symptom_index, precaution_dict

@st.cache_data(show_spinner=False)
def load_severity():
    severity_df = pd.read_csv("Symptom-severity.csv")
    severity_df.columns = severity_df.columns.str.strip().str.replace('\ufeff', '')
    return dict(zip(severity_df['Symptom'].str.lower().str.strip(), severity_df['weight']))

svm_model, nb_model, rf_model, encoder, symptom_index, precaution_dict = load_models()
symptom_severity = load_severity()

# ================== UI Setup =====================
st.set_page_config(page_title="🩺 Disease Predictor", page_icon="🧬", layout="centered")
st.title("🩺 Disease Prediction from Symptoms")

st.markdown("""
This app predicts potential diseases based on symptoms you select.
Choose symptoms from the dropdown below to get possible diagnoses and recommended precautions.
""")

# ================== Symptom Selection =====================
all_symptoms = sorted(symptom_index.keys())
display_symptoms = [s.replace("_", " ").title() for s in all_symptoms]
selected_display = st.multiselect("Select Symptoms:", display_symptoms)
selected_symptoms = [s.lower().replace(" ", "_") for s in selected_display]

# ================== Prediction Logic =====================
def predict_multiple_diseases(selected_symptoms):
    input_list = [0] * len(symptom_index)
    for s in selected_symptoms:
        if s in symptom_index:
            input_list[symptom_index[s]] = symptom_severity.get(s.replace("_", " "), 0)

    input_df = pd.DataFrame([input_list], columns=symptom_index.keys())

    preds = [
        encoder.classes_[rf_model.predict(input_df)[0]],
        encoder.classes_[nb_model.predict(input_df)[0]],
        encoder.classes_[svm_model.predict(input_df)[0]]
    ]

    top_preds = Counter(preds).most_common(3)
    result = []
    for disease, count in top_preds:
        precautions = precaution_dict.get(disease, ["Not available"] * 4)
        result.append({"Disease": disease, "Votes": count, "Precautions": precautions})

    return result

# ================== Output =====================
if st.button("🔍 Predict Disease"):
    if selected_symptoms:
        with st.spinner("Predicting based on selected symptoms..."):
            results = predict_multiple_diseases(selected_symptoms)
        for idx, res in enumerate(results):
            st.success(f"Prediction #{idx + 1}: {res['Disease']} ({res['Votes']} vote(s))")
            with st.expander("Precautions"):
                for p in res['Precautions']:
                    st.markdown(f"- {p}")
    else:
        st.warning("Please select at least one symptom.")

# ================== Footer =====================
st.markdown("""
---
✅ Made with ❤️ by [Sai Pawar](https://github.com/Saip2503)
""")
st.markdown("""
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/Saip2503)
""")