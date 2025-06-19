
# 💳 Fraud Detection App (ML-Powered)

This project is a **Machine Learning-powered web application** designed to detect fraudulent credit card transactions.
Built using Python and Streamlit, it provides users with a simple interface to upload transaction data and receive fraud predictions.

## 🚀 Live Demo
[🔗 Click here to try the live demo](https://fraud-transaction-detector.streamlit.app/) 

---

## 📌 Problem Statement

With the rise of digital banking and mobile money platforms in Nigeria, financial fraud has become a major concern. According to several reports, Nigeria loses billions of naira annually to fraudulent transactions, particularly in online banking and card-related fraud.

This project aims to build an intelligent fraud detection system that can automatically detect suspicious or fake transactions using machine learning techniques. The goal is to help banks, fintechs, and users quickly identify potential fraud and reduce financial loss.
This app leverages supervised machine learning (Logistic Regression and Random Forest) to identify potentially fraudulent transactions.

---

## 🧠 How It Works

1. **Pretrained ML Model**  
   A Random Forest model is trained on the famous [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It uses anonymized features (V1 to V28) and the `Amount` column to predict fraud.

2. **Streamlit Interface**  
   - Upload CSV or Excel files.
   - Model processes transactions.
   - Outputs:
     - Fraud prediction
     - Fraud probability
     - Option to download results.

---

## 🛠 Tech Stack

- **Python**
- **Pandas & NumPy** – Data manipulation
- **Scikit-learn** – Model training
- **Streamlit** – Web interface
- **Matplotlib & Seaborn** – Visualization
- **Joblib** – Model saving/loading

---

## 📂 Folder Structure

```
Fake-Bank-Transection-Detection-app/
│
├── app.py                       # Streamlit App
├── fraud Detection model.ipynb  # Notebook for Data exploration and model training
├── fraud_model.pkl              # Trained Random Forest model
├── app.py                       # Streamlit App
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Setup Instructions

### 📌 Clone and Run Locally

```bash
git clone https://github.com/busayo-I/Fake-Bank-Transaction-Detector.git
cd Fake-Bank-Transaction-Detector
```

### 🧪 Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # For Windows
# source venv/bin/activate  # For Mac/Linux
```

### 📦 Install Requirements

```bash
pip install -r requirements.txt
```

### ▶️ Run App

```bash
streamlit run app.py
```

---

## 🧪 Sample Test Data

To test the app:
- Upload a **CSV or Excel file** with columns:
  - `V1, V2, ..., V28, Amount`
- You can generate dummy data or extract samples from the original dataset on Kaggle.

---

## 📊 Model Performance

| Metric         | Value  |
|----------------|--------|
| Accuracy       | 99.9%  |
| Recall         | 90%    |
| Precision      | 99%    |
| AUC-ROC        | 0.9999 |

*Model tuned using StratifiedKFold Cross-validation & GridSearchCV.*

---

## ❗ Challenges Faced

- Handling class imbalance (fraud = 0.17%).
- Feature importance was hard to interpret due to anonymized features.
- Ensuring that prediction works smoothly with different user-uploaded file formats.

---

## 🙌 Contributors

- **Ibrahim Ismail Busayo** — Backend Engineer & Data scientist  
  _Built as part of the 3MTT Knowledge Showcase (June 2025)_

---

## 🔗 Useful Links

- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## 📄 License

This project is open-source and available under the MIT License.
