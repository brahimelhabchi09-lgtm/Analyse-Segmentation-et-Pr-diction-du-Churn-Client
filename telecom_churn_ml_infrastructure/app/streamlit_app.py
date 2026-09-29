import streamlit as st

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Telecom Customer Churn Prediction")
st.write("Infrastructure Streamlit prête pour le modèle de Machine Learning.")

st.info(
    "Ajoutez votre modèle entraîné dans le dossier models/ "
    "puis connectez-le à cette interface."
)

st.subheader("Infrastructure")
st.write("• Streamlit : application")
st.write("• MLflow : suivi des expériences")
st.write("• PostgreSQL : stockage des métadonnées MLflow")
st.write("• Docker : conteneurisation")
