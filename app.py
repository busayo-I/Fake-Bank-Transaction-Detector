import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Fraud Detection", layout="wide")

# Load model
model = joblib.load("fraud_detection_model.pkl")

# Title and styling
st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        h1 {color: #6a1b9a;}
        .stButton>button {background-color: #6a1b9a; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("🔍 AI-Powered Credit Card Fraud Detection")

st.write(" Upload a CSV or Excel file containing transection data with the features V1 to V28 and Amount to dectect whether the transaction are fraudulent or not.")
uploaded_file = st.file_uploader("Enter the chioce of file", type= ['csv', 'xlsx'])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)

        else:
            st.error("File not supported. Please upload a CSV or Excel File.")

        st.subheader("Data Preview")
        st.dataframe(data.head(10))
        st.write("Number of rows:", data.shape[0])
        st.write("Number of columns:", data.shape[1])

        required_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        if not all(col in data.columns for col in required_columns):
            st.error("The file does not contain all the columns required for the prediction. the columns should contain V1 to V28 and Amount.")
        else:
            x = data[required_columns]
            predictions = model.predict(x)
            probability = model.predict_proba(x)[:, 1]
            
            data['predictions'] = np.where(predictions == 1, "frauduent", "not fraudent")
            data['fraud probability (%)'] = (probability * 100).round(2)

            st.success("The prediction is successful!!!!!!")
            st.subheader(" This is the Prediction Result")
            st.dataframe(data[['predictions', 'fraud probability (%)']].value_counts().reset_index())

            st.subheader("This is the prediction result")

            st.dataframe(data.head(100))

            #download the result
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label = "Download Result",
                data = csv,
                file_name = "Fraud_Detection_ResultS.csv",
                mime = "text/csv"
            )
    except Exception as e:
        st.error(f"An error occurred during the file processing: {e}")

else:
    st.info("Please upload a CSV or Excel file to proceed with the fraud detection.")
    st.markdown("""
        <style>
            .stInfo {background-color: #e3f2fd; color: #0d47a1;}
        </style>
    """, unsafe_allow_html=True)


