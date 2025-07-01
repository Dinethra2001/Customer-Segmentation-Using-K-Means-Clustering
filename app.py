# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------- Data & model loading ----------
DATA_PATH  = "notebook/clustered_customers.csv"
SCALER_F   = "model/scaler.pkl"
KMEANS_F   = "model/kmeans.pkl"

@st.cache_data
def load_assets():
    df      = pd.read_csv(DATA_PATH)
    scaler  = joblib.load(SCALER_F)
    kmeans  = joblib.load(KMEANS_F)
    return df, scaler, kmeans

customer_df, scaler, kmeans = load_assets()

# ---------- Dashboard ----------
st.title("🧠 Customer Segmentation Dashboard")

st.subheader("Cluster distribution")
st.bar_chart(customer_df["Cluster"].value_counts())

st.subheader("Average features per cluster")
st.dataframe(
    customer_df
    .groupby("Cluster")[["Recency","Frequency","TotalUnits","TotalSpent"]]
    .mean()
    .round(2)
)

st.subheader("PCA scatter")
fig, ax = plt.subplots()
scatter = ax.scatter(
    customer_df["PCA1"], customer_df["PCA2"],
    c=customer_df["Cluster"], cmap="viridis", alpha=.7
)
ax.set_xlabel("PCA-1"); ax.set_ylabel("PCA-2")
st.pyplot(fig)

# ---------- (Optional) New-customer prediction form ----------
with st.expander("Predict cluster for a new customer"):
    recency   = st.number_input("Recency (days)",   min_value=0)
    frequency = st.number_input("Frequency",        min_value=0)
    units     = st.number_input("Total units",      min_value=0)
    spent     = st.number_input("Total spent (£)",  min_value=0.0, format="%.2f")
    if st.button("Predict"):
        X_new   = [[recency, frequency, units, spent]]
        X_scaled = scaler.transform(X_new)
        label    = int(kmeans.predict(X_scaled)[0])
        st.success(f"Predicted cluster ➜ **{label}**")
