"""Client segmentation without using the churn target as an input."""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMNS = {"churn"}
IDENTIFIER_COLUMNS = {"customerid", "customer_id"}
SILHOUETTE_SAMPLE_SIZE = 2000


def prepare_customer_features(df):
    """Encode and scale customer attributes, excluding target and identifiers."""
    excluded_columns = {
        column
        for column in df.columns
        if str(column).strip().casefold() in TARGET_COLUMNS | IDENTIFIER_COLUMNS
    }
    features = df.drop(columns=list(excluded_columns)).copy()
    features = features.dropna(axis=1, how="all")

    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("Aucune caractéristique client exploitable n'a été trouvée.")

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers)
    matrix = preprocessor.fit_transform(features)

    if not np.isfinite(matrix).all():
        raise ValueError("Les caractéristiques préparées contiennent des valeurs non finies.")

    return features, matrix


def _cluster_metrics(matrix, labels, ignore_noise=False):
    labels = np.asarray(labels)
    valid = labels != -1 if ignore_noise else np.ones(labels.shape, dtype=bool)
    valid_labels = labels[valid]
    unique_labels = np.unique(valid_labels)

    metrics = {
        "n_clusters": int(len(unique_labels)),
        "n_noise": int(np.sum(labels == -1)) if ignore_noise else 0,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if len(unique_labels) < 2 or len(unique_labels) >= len(valid_labels):
        return metrics

    valid_matrix = matrix[valid]
    metrics["silhouette"] = float(
        silhouette_score(
            valid_matrix,
            valid_labels,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(valid_labels)),
            random_state=42,
        )
    )
    metrics["davies_bouldin"] = float(
        davies_bouldin_score(valid_matrix, valid_labels)
    )
    metrics["calinski_harabasz"] = float(
        calinski_harabasz_score(valid_matrix, valid_labels)
    )
    return metrics


def _cluster_profiles(features, labels):
    profile_data = features.copy()
    profile_data["cluster"] = labels
    numeric_columns = profile_data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    profile_rows = []
    for cluster, group in profile_data.groupby("cluster", sort=True):
        row = {"cluster": cluster, "customer_count": len(group)}
        for column in numeric_columns:
            row[f"{column}_mean"] = group[column].mean()
        for column in categorical_columns:
            modes = group[column].mode(dropna=True)
            row[f"{column}_mode"] = modes.iloc[0] if not modes.empty else None
        profile_rows.append(row)

    return pd.DataFrame(profile_rows)


def _pca_coordinates(matrix, labels):
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("La PCA nécessite au moins deux clients et deux caractéristiques.")

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(matrix)
    projection = pd.DataFrame(coordinates, columns=["PC1", "PC2"])
    projection["cluster"] = labels
    return projection, pca.explained_variance_ratio_


def cluster_customers(
    df,
    algorithm="kmeans",
    max_k=10,
    n_clusters=4,
    eps=0.5,
    min_samples=5,
    random_state=42,
):
    
    features, matrix = prepare_customer_features(df)
    sample_count = matrix.shape[0]
    if sample_count < 3:
        raise ValueError("Le clustering nécessite au moins trois clients.")

    algorithm = algorithm.casefold()
    elbow = pd.DataFrame(columns=["k", "inertia", "silhouette"])

    if algorithm == "kmeans":
        max_candidate = min(max(2, int(max_k)), sample_count - 1)
        candidates = []
        for k in range(1, max_candidate + 1):
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            candidate_labels = model.fit_predict(matrix)
            score = (
                float(
                    silhouette_score(
                        matrix,
                        candidate_labels,
                        sample_size=min(SILHOUETTE_SAMPLE_SIZE, sample_count),
                        random_state=random_state,
                    )
                )
                if k > 1
                else None
            )
            candidates.append(
                {"k": k, "inertia": float(model.inertia_), "silhouette": score}
            )
        elbow = pd.DataFrame(candidates)
        best_row = elbow.loc[elbow["silhouette"].idxmax()]
        selected_k = int(best_row["k"])
        model = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
        labels = model.fit_predict(matrix)
    elif algorithm == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(matrix)
    elif algorithm in {"agglomerative", "agglomerative clustering"}:
        selected_k = min(max(2, int(n_clusters)), sample_count - 1)
        model = AgglomerativeClustering(n_clusters=selected_k)
        labels = model.fit_predict(matrix)
    else:
        raise ValueError("algorithm doit être 'kmeans', 'dbscan' ou 'agglomerative'.")

    metrics = _cluster_metrics(
        matrix,
        labels,
        ignore_noise=(algorithm == "dbscan"),
    )
    profiles = _cluster_profiles(features, labels)
    projection, explained_variance = _pca_coordinates(matrix, labels)

    assignments = df.copy()
    assignments["cluster"] = labels
    return {
        "assignments": assignments,
        "metrics": metrics,
        "profiles": profiles,
        "elbow": elbow,
        "pca": projection,
        "pca_explained_variance": explained_variance,
        "feature_columns": features.columns.tolist(),
        "algorithm": algorithm,
    }