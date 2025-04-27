
import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter

import pandas as pd
import numpy as np
import pickle

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

# --- Prediction Function ---
def predict_multiple_diseases(input_symptoms):
    input_list = [0] * len(symptom_index)
    symptom_list = [s.strip().lower().replace(" ", "_") for s in input_symptoms.split(",")]

    for s in symptom_list:
        if s in symptom_index:
            input_list[symptom_index[s]] = symptom_severity.get(s.replace("_", " "), 0)

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

# --- Streamlit UI ---
st.set_page_config(page_title="Disease Prediction from Symptoms", page_icon="🩺", layout="wide")

st.title("🩺 Disease Prediction System")
st.markdown("Enter your symptoms below, separated by commas.")

symptom_input = st.text_input("Symptoms (comma separated)", placeholder="e.g., itching, skin rash, nodal skin eruptions")

if st.button("Predict"):
    if symptom_input:
        results = predict_multiple_diseases(symptom_input)

        for idx, res in enumerate(results):
            st.subheader(f"Prediction #{idx+1}: {res['Disease']}")
            st.write(f"Votes: {res['Votes']} / 3")
            st.markdown("**Recommended Precautions:**")
            for p in res["Precautions"]:
                st.write(f"- {p}")
    else:
        st.warning("⚠️ Please enter some symptoms.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
