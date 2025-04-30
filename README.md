# 🚨 Multi-Disease Prediction using Machine Learning

This project is a machine learning-based disease prediction system that takes user-provided symptoms and predicts the most likely diseases using an ensemble of classifiers. The app also provides recommended precautions for the predicted diseases.
My Streamlit App: https://diseaseprediction25.streamlit.app/
## 🚀 Live Demo

You can run the web application using Streamlit:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
.
🔗 app.py                      # Streamlit web app
📄 dataset.csv                # Main dataset with diseases and symptoms
📄 Symptom-severity.csv       # Symptom severity weights
📄 symptom_Description.csv    # Description of symptoms (optional)
📄 symptom_precaution.csv     # Disease-wise precautions
📄 rf_model.pkl               # Trained Random Forest model
📄 nb_model.pkl               # Trained Naive Bayes model
📄 svm_model.pkl              # Trained SVM model
📄 encoder.pkl                # Label encoder for disease labels
📄 symptom_index.pkl          # Mapping of symptom to index
📄 precaution_dict.pkl        # Mapping of diseases to precautions
📄 requirements.txt           # Python dependencies
📄 README.md                  # Project documentation
```

---

## 💡 Features

- Predict diseases from a list of symptoms using an ensemble of:
  - Random Forest
  - Naive Bayes
  - Support Vector Machine
- Displays top 3 disease predictions based on majority voting.
- Shows recommended precautions for each disease.
- Clean and interactive UI built with Streamlit.

---

## 🧠 Models and Accuracy

The project uses three machine learning models trained on symptom presence (with one-hot encoding) and severity:

- **Random Forest**
- **Naive Bayes**
- **Support Vector Machine**

Each model is evaluated with classification reports and confusion matrices. Random oversampling was applied to handle class imbalance.

---

## 🛠️ Installation

### Clone the Repository

```bash
git clone https://github.com/Saip2503/multi-disease-prediction.git
cd multi-disease-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🥺 How to Use

1. Run the app:

   ```bash
   streamlit run app.py
   ```

2. Open your browser to the local Streamlit server (usually http://localhost:8501).

3. Select one or more symptoms from the dropdown.

4. Click on **Predict Disease** to get results.

---

## 📊 Dataset Source

- [Disease-Symptom-Description Dataset (Kaggle)](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

---

## 📝 Requirements

```
streamlit
pandas
numpy
scikit-learn
imblearn
matplotlib
seaborn
```

---

## 🙇‍♂️ Author

**Sai Sunil Pawar**  
B.Tech in Electronics and Computer Science (AI/ML & Cloud), Mumbai University  
GitHub: [@Saip2503](https://github.com/Saip2503)

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

