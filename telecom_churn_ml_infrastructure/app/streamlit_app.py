from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.clustering import cluster_customers
from src.classification import compare_churn_models

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Telecom Customer Churn Prediction")
data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "telecom_churn.csv"
uploaded_file = st.file_uploader("Charger un fichier CSV", type="csv")
if uploaded_file is not None:
    customers = pd.read_csv(uploaded_file)
elif data_path.exists():
    customers = pd.read_csv(data_path)
else:
    st.warning("Chargez un fichier CSV pour commencer.")
    st.stop()


def render_clustering(customers):
    st.header("Segmentation des clients")
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


def render_classification(customers):
    st.header("Prédiction du churn")
    st.caption(
        "Les cinq modèles utilisent le même découpage stratifié. "
        "Imputation, standardisation et suréchantillonnage sont ajustés sur l'entraînement uniquement."
    )

    clustering_result = st.session_state.get("clustering_result")
    cluster_available = False
    if clustering_result is not None:
        clustered_customers = clustering_result["assignments"].drop(
            columns=["cluster"], errors="ignore"
        )
        cluster_available = clustered_customers.reset_index(drop=True).equals(
            customers.reset_index(drop=True)
        )
        if not cluster_available:
            st.warning(
                "Le résultat de segmentation ne correspond pas au fichier courant. "
                "Relancez la segmentation pour comparer avec le cluster."
            )

    include_cluster = st.checkbox(
        "Comparer aussi avec les variables initiales + cluster",
        value=cluster_available,
        disabled=not cluster_available,
    )
    imbalance_method = st.selectbox(
        "Gestion du déséquilibre des classes",
        ["Suréchantillonnage aléatoire", "Aucune"],
    )
    method = "random_oversampling" if imbalance_method == "Suréchantillonnage aléatoire" else "none"
    test_size = st.slider("Part réservée au test", 0.1, 0.4, 0.2, 0.05)

    if st.button("Entraîner et comparer les modèles", type="primary"):
        model_data = customers.copy()
        cluster_column = None
        if include_cluster and cluster_available:
            model_data["cluster"] = clustering_result["assignments"]["cluster"].to_numpy()
            cluster_column = "cluster"
        with st.spinner("Entraînement et évaluation des modèles..."):
            try:
                st.session_state["classification_result"] = compare_churn_models(
                    model_data,
                    cluster_column=cluster_column,
                    test_size=test_size,
                    imbalance_method=method,
                )
                st.session_state["classification_has_cluster"] = cluster_column is not None
                st.session_state["classification_source"] = customers.reset_index(drop=True)
            except (ValueError, TypeError) as error:
                st.error(f"Impossible d'entraîner les modèles : {error}")

    result = st.session_state.get("classification_result")
    result_source = st.session_state.get("classification_source")
    if result is not None and (
        result_source is None
        or not result_source.equals(customers.reset_index(drop=True))
    ):
        result = None
    if result is None:
        return

    metrics = result["metrics"]
    st.subheader("Performances sur le jeu de test")
    st.dataframe(
        metrics.sort_values(["Variables", "F1-score"], ascending=[True, False]).style.format(
            {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    if st.session_state.get("classification_has_cluster"):
        f1_comparison = metrics.pivot(
            index="Modèle", columns="Variables", values="F1-score"
        )
        auc_comparison = metrics.pivot(
            index="Modèle", columns="Variables", values="ROC-AUC"
        )
        expected_sets = {"Variables initiales", "Variables initiales + cluster"}
        if expected_sets.issubset(f1_comparison.columns) and expected_sets.issubset(
            auc_comparison.columns
        ):
            comparison = pd.DataFrame(
                {
                    "Gain F1 avec cluster": (
                        f1_comparison["Variables initiales + cluster"]
                        - f1_comparison["Variables initiales"]
                    ),
                    "Gain ROC-AUC avec cluster": (
                        auc_comparison["Variables initiales + cluster"]
                        - auc_comparison["Variables initiales"]
                    ),
                }
            )
            st.subheader("Effet de la segmentation")
            st.dataframe(
                comparison.sort_values("Gain F1 avec cluster", ascending=False).style.format("{:.3f}"),
                use_container_width=True,
            )

    st.subheader("Matrices de confusion")
    model_names = metrics["Modèle"].drop_duplicates().tolist()
    selected_model = st.selectbox("Modèle à examiner", model_names)
    selected_matrices = [
        (feature_set, result["confusion_matrices"][(selected_model, feature_set)])
        for feature_set in metrics.loc[metrics["Modèle"] == selected_model, "Variables"].unique()
    ]
    matrix_columns = st.columns(len(selected_matrices))
    for column, (feature_set, matrix) in zip(matrix_columns, selected_matrices):
        with column:
            st.caption(feature_set)
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Réel : non-churn", "Réel : churn"],
                    columns=["Prédit : non-churn", "Prédit : churn"],
                ),
                use_container_width=True,
            )


segmentation_tab, classification_tab = st.tabs(["Segmentation", "Classification"])
with segmentation_tab:
    render_clustering(customers)
with classification_tab:
    render_classification(customers)
