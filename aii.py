import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter

# Fungsi untuk menghitung entropi
def entropy(y):
    total = len(y)
    counts = Counter(y)
    probabilities = [count / total for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# Fungsi Information Gain
def information_gain(data, feature, target):
    entropy_before = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset = data[data[feature] == value]
        weighted_entropy += (count / len(data)) * entropy(subset[target])
    return entropy_before - weighted_entropy

# Streamlit interface
st.title("E-commerce Customer Behavior Prediction")
st.write("Upload dataset dan pilih fitur untuk memprediksi.")

# File uploader
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.write(data.head())
    
    # Label Encoding
    st.write("Melakukan Label Encoding untuk fitur kategorikal.")
    categorical_cols = data.select_dtypes(include=["object", "bool"]).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    st.write("Data setelah encoding:")
    st.write(data.head())

    # Information Gain
    st.write("Menghitung Information Gain untuk setiap fitur.")
    target = st.selectbox("Pilih kolom target:", data.columns)
    features = [col for col in data.columns if col != target]
    ig_results = {feature: information_gain(data, feature, target) for feature in features}
    st.write("Hasil Information Gain:")
    st.write(pd.DataFrame(ig_results.items(), columns=["Fitur", "Information Gain"]).sort_values(by="Information Gain", ascending=False))

    # Memuat model dan prediksi
    st.write("Prediksi dengan Random Forest.")
    model_file = "random_forest_model_90.pkl"
    try:
        model = joblib.load(model_file)
        selected_features = st.multiselect("Pilih fitur untuk prediksi:", features)
        if st.button("Prediksi"):
            if set(selected_features).issubset(data.columns):
                predictions = model.predict(data[selected_features])
                data["Prediction"] = predictions
                st.write("Hasil prediksi:")
                st.write(data.head())
                st.download_button("Download Hasil Prediksi", data.to_csv(index=False), file_name="predictions.csv")
            else:
                st.error("Fitur yang dipilih tidak tersedia dalam dataset.")
    except FileNotFoundError:
        st.error(f"Model {model_file} tidak ditemukan. Harap unggah model terlebih dahulu.")
