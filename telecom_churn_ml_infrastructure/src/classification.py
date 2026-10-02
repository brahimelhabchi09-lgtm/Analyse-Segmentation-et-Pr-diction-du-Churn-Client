"""Churn classification with a leakage-safe model comparison workflow."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN = "churn"
IDENTIFIER_COLUMNS = {"customerid", "customer_id"}


def _find_target_column(df):
    matches = [column for column in df.columns if str(column).strip().casefold() == TARGET_COLUMN]
    if len(matches) != 1:
        raise ValueError("Le fichier doit contenir une unique colonne cible 'Churn'.")
    return matches[0]


def _encode_target(target):
    values = target.astype(str).str.strip().str.casefold()
    classes = values.unique().tolist()
    if len(classes) != 2:
        raise ValueError("La cible churn doit contenir exactement deux classes non vides.")

    positive_tokens = {"yes", "true", "1", "churn", "churned"}
    positive_classes = [value for value in classes if value in positive_tokens]
    if positive_classes:
        positive_class = positive_classes[0]
    else:
        positive_class = sorted(classes)[-1]
    return (values == positive_class).astype(int).to_numpy()


def _prepare_features(features, cluster_column=None, reference_features=None):
    prepared = features.copy()
    reference = features if reference_features is None else reference_features
    numeric_cluster = cluster_column if cluster_column in prepared.columns else None
    if numeric_cluster is not None:
        prepared[numeric_cluster] = prepared[numeric_cluster].astype("string")

    for column in reference.select_dtypes(include=["object", "string"]).columns:
        if column == numeric_cluster:
            continue
        stripped = reference[column].astype("string").str.strip()
        nonempty = stripped.notna() & stripped.ne("")
        converted = pd.to_numeric(stripped, errors="coerce")
        if nonempty.any() and converted[nonempty].notna().all():
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    empty_columns = reference.columns[reference.isna().all()]
    prepared = prepared.drop(columns=empty_columns)
    numeric_columns = prepared.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in prepared.columns if column not in numeric_columns
    ]
    if not numeric_columns and not categorical_columns:
        raise ValueError("Aucune variable prédictive exploitable n'a été trouvée.")

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    return prepared, ColumnTransformer(transformers=transformers)


def _model_factories(random_state):
    return {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=random_state),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        "SVM": lambda: SVC(probability=True, random_state=random_state),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            random_state=random_state
        ),
    }


def compare_churn_models(
    df,
    cluster_column=None,
    test_size=0.2,
    random_state=42,
    imbalance_method="random_oversampling",
):
    """Compare five classifiers on the same split, with and without a cluster feature."""
    if imbalance_method not in {"random_oversampling", "none"}:
        raise ValueError("imbalance_method doit être 'random_oversampling' ou 'none'.")
    if not 0 < test_size < 1:
        raise ValueError("test_size doit être compris entre 0 et 1.")

    target_column = _find_target_column(df)
    usable = df.loc[df[target_column].notna()].copy()
    y = _encode_target(usable[target_column])
    excluded = IDENTIFIER_COLUMNS | {TARGET_COLUMN}
    feature_columns = [
        column
        for column in usable.columns
        if str(column).strip().casefold() not in excluded
        and column != cluster_column
    ]
    if not feature_columns:
        raise ValueError("Aucune variable initiale n'est disponible pour la classification.")
    if cluster_column is not None and cluster_column not in usable.columns:
        raise ValueError(f"La colonne de cluster '{cluster_column}' est absente du fichier.")

    class_counts = np.bincount(y, minlength=2)
    if class_counts.min() < 2:
        raise ValueError("Chaque classe doit contenir au moins deux clients.")

    indices = np.arange(len(usable))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    model_factories = _model_factories(random_state)
    metrics_rows = []
    confusion_matrices = {}
    fitted_models = {}

    feature_sets = [("Variables initiales", feature_columns)]
    if cluster_column is not None:
        feature_sets.append(
            ("Variables initiales + cluster", feature_columns + [cluster_column])
        )

    for feature_set_name, selected_columns in feature_sets:
        raw_features = usable[selected_columns]
        prepared, preprocessor = _prepare_features(
            raw_features,
            cluster_column=cluster_column if cluster_column in selected_columns else None,
            reference_features=raw_features.iloc[train_indices],
        )
        for model_name, make_model in model_factories.items():
            sampler = (
                RandomOverSampler(random_state=random_state)
                if imbalance_method == "random_oversampling"
                else "passthrough"
            )
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("sampler", sampler),
                    ("classifier", make_model()),
                ]
            )
            pipeline.fit(prepared.iloc[train_indices], y[train_indices])
            predictions = pipeline.predict(prepared.iloc[test_indices])
            probabilities = pipeline.predict_proba(prepared.iloc[test_indices])[:, 1]
            matrix = confusion_matrix(y[test_indices], predictions, labels=[0, 1])
            metrics_rows.append(
                {
                    "Modèle": model_name,
                    "Variables": feature_set_name,
                    "Precision": precision_score(
                        y[test_indices], predictions, zero_division=0
                    ),
                    "Recall": recall_score(y[test_indices], predictions, zero_division=0),
                    "F1-score": f1_score(y[test_indices], predictions, zero_division=0),
                    "ROC-AUC": roc_auc_score(y[test_indices], probabilities),
                }
            )
            confusion_matrices[(model_name, feature_set_name)] = matrix
            fitted_models[(model_name, feature_set_name)] = pipeline

    return {
        "metrics": pd.DataFrame(metrics_rows),
        "confusion_matrices": confusion_matrices,
        "models": fitted_models,
        "test_size": test_size,
        "imbalance_method": imbalance_method,
        "target_column": target_column,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }