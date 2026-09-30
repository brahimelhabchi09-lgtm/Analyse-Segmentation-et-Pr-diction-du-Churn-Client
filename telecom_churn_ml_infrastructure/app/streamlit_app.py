from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.clustering import cluster_customers

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


def render_clustering():
    st.header("Segmentation des clients")
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "telecom_churn.csv"
    uploaded_file = st.file_uploader("Charger un fichier CSV", type="csv")

    if uploaded_file is not None:
        customers = pd.read_csv(uploaded_file)
    elif data_path.exists():
        customers = pd.read_csv(data_path)
    else:
        st.warning("Chargez un fichier CSV pour commencer.")
        return

    st.caption(
        f"{len(customers):,} clients. La cible churn et les identifiants sont exclus des variables de clustering."
    )
    algorithm_name = st.selectbox(
        "Algorithme",
        ["K-Means", "DBSCAN", "Agglomerative Clustering"],
    )
    algorithm = {
        "K-Means": "kmeans",
        "DBSCAN": "dbscan",
        "Agglomerative Clustering": "agglomerative",
    }[algorithm_name]

    options = {}
    if algorithm == "kmeans":
        options["max_k"] = st.slider("Nombre maximal de clusters à tester", 2, 10, 8)
    elif algorithm == "agglomerative":
        options["n_clusters"] = st.slider("Nombre de clusters", 2, 10, 4)
    else:
        options["eps"] = st.slider("Epsilon", 0.1, 3.0, 0.5, 0.1)
        options["min_samples"] = st.slider("Échantillons minimum", 2, 20, 5)

    if st.button("Lancer la segmentation", type="primary"):
        with st.spinner("Calcul des clusters..."):
            try:
                st.session_state["clustering_result"] = cluster_customers(
                    customers,
                    algorithm=algorithm,
                    **options,
                )
                st.session_state["clustering_algorithm"] = algorithm_name
            except (ValueError, TypeError) as error:
                st.error(f"Impossible de calculer les clusters : {error}")

    result = st.session_state.get("clustering_result")
    if result is None:
        return

    st.subheader(f"Résultats : {st.session_state['clustering_algorithm']}")
    metrics = result["metrics"]
    score_columns = st.columns(4)
    score_columns[0].metric("Clusters", metrics["n_clusters"])
    score_columns[1].metric(
        "Silhouette",
        "N/A" if metrics["silhouette"] is None else f"{metrics['silhouette']:.3f}",
    )
    score_columns[2].metric(
        "Davies-Bouldin",
        "N/A" if metrics["davies_bouldin"] is None else f"{metrics['davies_bouldin']:.3f}",
    )
    score_columns[3].metric(
        "Calinski-Harabasz",
        "N/A" if metrics["calinski_harabasz"] is None else f"{metrics['calinski_harabasz']:.1f}",
    )

    if metrics["n_noise"]:
        st.caption(f"Points considérés comme bruit par DBSCAN : {metrics['n_noise']}")

    if not result["elbow"].empty:
        st.subheader("Choix du nombre de clusters")
        elbow_tab, silhouette_tab = st.tabs(["Elbow (inertie)", "Silhouette"])
        with elbow_tab:
            st.line_chart(result["elbow"].set_index("k")["inertia"])
        with silhouette_tab:
            silhouette_data = result["elbow"].dropna(subset=["silhouette"])
            st.line_chart(silhouette_data.set_index("k")["silhouette"])

    st.subheader("Profils des clients")
    st.dataframe(result["profiles"], use_container_width=True, hide_index=True)

    st.subheader("Projection PCA")
    variance = result["pca_explained_variance"]
    pca_chart = px.scatter(
        result["pca"],
        x="PC1",
        y="PC2",
        color=result["pca"]["cluster"].astype(str),
        labels={"color": "Cluster"},
        title=(
            f"Variance expliquée : PC1 {variance[0]:.1%}, "
            f"PC2 {variance[1]:.1%}"
        ),
    )
    st.plotly_chart(pca_chart, use_container_width=True)

    csv_data = result["assignments"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger les clients avec leur cluster",
        data=csv_data,
        file_name="clients_segmentes.csv",
        mime="text/csv",
    )


render_clustering()
