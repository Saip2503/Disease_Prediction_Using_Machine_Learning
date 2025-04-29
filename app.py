import streamlit as st
import pandas as pd
import pickle
from collections import Counter
import numpy as np

# ================== Load Models and Data =====================

# Load symptom severity
severity_df = pd.read_csv("Symptom-severity.csv")
severity_df.columns = severity_df.columns.str.strip().str.replace('\ufeff', '')
symptom_severity = dict(zip(severity_df['Symptom'].str.lower().str.strip(), severity_df['weight']))

# Load precautions
precaution_df = pd.read_csv("symptom_precaution.csv").fillna("")
precaution_df.columns = precaution_df.columns.str.strip().str.replace('\ufeff', '')
precaution_dict = {
    row["Disease"]: [row["Precaution_1"], row["Precaution_2"], row["Precaution_3"], row["Precaution_4"]]
    for _, row in precaution_df.iterrows()
}

# Load the trained models and encoder
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

# ================== Streamlit UI =====================

st.set_page_config(page_title="Disease Prediction App", page_icon="🩺", layout="centered")
st.title("🩺 **Disease Prediction from Symptoms**")

# Description of the app
st.markdown("""
    This app helps in predicting potential diseases based on the symptoms you provide.
    You can select multiple symptoms from the dropdown, and the app will return the top diseases along with recommended precautions.
""")

# Dropdown list for symptom selection
# Assuming 'all_symptoms' is the list of all available symptoms from the dataset
all_symptoms = sorted([
    'itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 'shivering',
    'chills', 'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting',
    'vomiting', 'burning micturition', 'spotting urination', 'fatigue', 'weight gain',
    'anxiety', 'cold hands and feets', 'mood swings', 'weight loss', 'restlessness',
    'lethargy', 'patches in throat', 'irregular sugar level', 'cough', 'high fever',
    'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion'
])

selected_symptoms = st.multiselect(
    "Select Symptoms (Multiple can be selected):", 
    all_symptoms, 
    key="symptoms"
)

# Prediction function
def predict_multiple_diseases(selected_symptoms):
    input_list = [0] * len(symptom_index)
    symptom_list = [s.strip().lower().replace(" ", "_") for s in selected_symptoms]

    for s in symptom_list:
        if s in symptom_index:
            input_list[symptom_index[s]] = symptom_severity.get(s.replace("_", " "), 0)

    input_df = pd.DataFrame([input_list], columns=symptom_index.keys())

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

# Prediction button
if st.button("Predict Disease"):
    if selected_symptoms:
        with st.spinner("Predicting..."):
            results = predict_multiple_diseases(selected_symptoms)

        # Display predictions
        for idx, res in enumerate(results):
            st.subheader(f"Prediction #{idx + 1}: {res['Disease']}")
            st.write(f"Votes: {res['Votes']} / 3")
            st.markdown("**Precautions:**")
            for p in res['Precautions']:
                st.write(f"- {p}")
    else:
        st.warning("Please select at least one symptom!")

# ================== Footer =====================

# Add Footer
st.markdown(
    """
    ---
    Made with ❤️ by [Sai Pawar](https://github.com/Saip2503)
    """
)
