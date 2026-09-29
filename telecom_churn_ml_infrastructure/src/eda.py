import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CHARGER DATASET
# ============================================================

def load_data(file_path):
    # كنقراو fichier CSV وكنحطوه داخل DataFrame
    df = pd.read_csv(file_path)

    print(f"Dataset chargé : {file_path}")
    print(f"Nombre de lignes : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")

    return df


# ============================================================
# 2. NETTOYAGE DES TYPES
# ============================================================

def convert_data_types(df):
    # TotalCharges خاصها تكون numérique
    # ولكن فـ raw data جاية object
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    print("\n===== TYPES APRÈS CONVERSION =====")
    print(df.dtypes)

    return df


# ============================================================
# 3. INFORMATIONS GÉNÉRALES
# ============================================================

def show_basic_information(df):
    # shape كتعطينا عدد lignes و columns
    print("\n===== SHAPE =====")
    print(df.shape)

    # كنشوفو أسماء الأعمدة
    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    # كنشوفو types ديال الأعمدة
    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    # info() كتعطينا معلومات عامة
    print("\n===== INFO =====")
    df.info()


# ============================================================
# 4. STATISTIQUES
# ============================================================

def show_statistics(df):
    # statistiques ديال variables numériques
    print("\n===== STATISTIQUES NUMÉRIQUES =====")
    print(df.describe())

    # statistiques ديال جميع variables
    print("\n===== STATISTIQUES COMPLETES =====")
    print(df.describe(include="all"))


# ============================================================
# 5. VALEURS MANQUANTES
# ============================================================

def check_missing_values(df):
    # كنحسبو عدد القيم الناقصة
    missing = df.isnull().sum()

    # كنحسبو النسبة المئوية
    missing_percent = (
        df.isnull().sum() / len(df) * 100
    ).round(2)

    # كنصايبو DataFrame فيه النتائج
    result = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_percent
    })

    # كنخليو غير الأعمدة اللي فيهم missing
    result = result[
        result["missing_count"] > 0
    ]

    result = result.sort_values(
        by="missing_count",
        ascending=False
    )

    print("\n===== VALEURS MANQUANTES =====")

    if result.empty:
        print("Aucune valeur manquante.")

    else:
        print(result)

    return result


def plot_missing_values(df, save_path=None):
    # كنحسبو missing values
    missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    if missing.empty:
        print("Aucune valeur manquante.")
        return

    # كنرسمو graphique
    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=missing.index,
        y=missing.values
    )

    plt.title("Valeurs manquantes")
    plt.xlabel("Variables")
    plt.ylabel("Nombre")
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


# ============================================================
# 6. DOUBLONS
# ============================================================

def check_duplicates(df):
    # كنشوفو واش كاينين lignes مكررين
    duplicates = df.duplicated().sum()

    print("\n===== DOUBLONS =====")
    print(f"Nombre de doublons : {duplicates}")

    return duplicates


def remove_duplicates(df):
    # كنحسبو الحجم قبل الحذف
    before = len(df)

    # كنحيدو doublons
    df = df.drop_duplicates()

    # كنعاودو نرتبو index
    df = df.reset_index(drop=True)

    # الحجم من بعد الحذف
    after = len(df)

    print(f"Lignes avant : {before}")
    print(f"Lignes après : {after}")
    print(f"Doublons supprimés : {before - after}")

    return df


# ============================================================
# 7. SUPPRIMER IDENTIFIANT
# ============================================================

def remove_identifier(df):
    # customerID غير identifier
    # ما عندها حتى معنى بالنسبة للموديل
    # لذلك غادي نحيدوها
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

        print("\ncustomerID supprimée.")

    return df


# ============================================================
# 8. VARIABLES NUMÉRIQUES
# ============================================================

def get_numeric_columns(df):
    # كنجيبو جميع variables numériques
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Churn ماشي feature
    # إلا كانت داخلة نحيدوها
    if "Churn" in numeric_columns:
        numeric_columns.remove("Churn")

    return numeric_columns


# ============================================================
# 9. VARIABLES CATÉGORIELLES
# ============================================================

def get_categorical_columns(df):
    # كنجيبو variables catégorielles
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Churn هي target
    # لذلك ما خاصهاش تكون ضمن features
    if "Churn" in categorical_columns:
        categorical_columns.remove("Churn")

    return categorical_columns


def show_column_types(df):
    # كنجيبو numeric variables
    numeric_columns = get_numeric_columns(df)

    # كنجيبو categorical variables
    categorical_columns = get_categorical_columns(df)

    print("\n===== VARIABLES NUMÉRIQUES =====")
    print(numeric_columns)

    print("\n===== VARIABLES CATÉGORIELLES =====")
    print(categorical_columns)

    return numeric_columns, categorical_columns


# ============================================================
# 10. DISTRIBUTION VARIABLES NUMÉRIQUES
# ============================================================

def plot_numeric_distributions(
    df,
    numeric_columns
):
    # كنرسمو distribution ديال كل variable numérique
    for column in numeric_columns:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=column,
            kde=True
        )

        plt.title(
            f"Distribution de {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Fréquence")

        plt.tight_layout()
        plt.show()


# ============================================================
# 11. BOXPLOTS
# ============================================================

def plot_boxplots(
    df,
    numeric_columns
):
    # Boxplot كيساعدنا نكتاشفو outliers
    for column in numeric_columns:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            data=df,
            x=column
        )

        plt.title(
            f"Boxplot - {column}"
        )

        plt.xlabel(column)

        plt.tight_layout()
        plt.show()


# ============================================================
# 12. DETECTION OUTLIERS IQR
# ============================================================

def detect_outliers_iqr(
    df,
    numeric_columns
):
    # هنا غادي نجمعو نتائج outliers
    results = []

    for column in numeric_columns:

        # Q1 = 25%
        q1 = df[column].quantile(0.25)

        # Q3 = 75%
        q3 = df[column].quantile(0.75)

        # IQR = Q3 - Q1
        iqr = q3 - q1

        # الحد السفلي
        lower_bound = q1 - 1.5 * iqr

        # الحد العلوي
        upper_bound = q3 + 1.5 * iqr

        # القيم اللي خارج الحدود
        outliers = df[
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ]

        results.append({
            "variable": column,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers_count": len(outliers),
            "outliers_percent": round(
                len(outliers) / len(df) * 100,
                2
            )
        })

    result = pd.DataFrame(results)

    print("\n===== OUTLIERS IQR =====")
    print(result)

    return result


# ============================================================
# 13. DISTRIBUTION VARIABLES CATÉGORIELLES
# ============================================================

def plot_categorical_distributions(
    df,
    categorical_columns
):
    # كنرسمو عدد clients فكل catégorie
    for column in categorical_columns:

        plt.figure(figsize=(10, 5))

        sns.countplot(
            data=df,
            x=column
        )

        plt.title(
            f"Distribution de {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Nombre")

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


# ============================================================
# 14. MATRICE CORRÉLATION
# ============================================================

def plot_correlation_matrix(
    df,
    numeric_columns
):
    # كنحسبو correlation
    correlation = df[numeric_columns].corr()

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Matrice de corrélation"
    )

    plt.tight_layout()
    plt.show()

    return correlation


# ============================================================
# 15. ANALYSE CHURN
# ============================================================

def analyze_target(
    df,
    target="Churn"
):
    # Churn هي target ديال classification
    print(
        f"\n===== TARGET : {target} ====="
    )

    # عدد clients فكل classe
    print("\nNombre par classe :")

    print(
        df[target].value_counts()
    )

    # النسبة المئوية لكل classe
    print("\nPourcentage par classe :")

    print(
        (
            df[target]
            .value_counts(
                normalize=True
            ) * 100
        ).round(2)
    )

    # كنرسمو distribution
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x=target
    )

    plt.title(
        "Distribution du Churn"
    )

    plt.xlabel("Churn")
    plt.ylabel("Nombre de clients")

    plt.show()


# ============================================================
# 16. VARIABLES NUMÉRIQUES VS CHURN
# ============================================================

def analyze_numeric_vs_target(
    df,
    numeric_columns,
    target="Churn"
):
    # كنشوفو كيفاش variables numériques
    # كيتوزعو حسب Churn
    for column in numeric_columns:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=target,
            y=column
        )

        plt.title(
            f"{column} selon {target}"
        )

        plt.xlabel(target)
        plt.ylabel(column)

        plt.tight_layout()
        plt.show()


# ============================================================
# 17. VARIABLES CATÉGORIELLES VS CHURN
# ============================================================

def analyze_categorical_vs_target(
    df,
    categorical_columns,
    target="Churn"
):
    # كنقارن كل variable catégorielle مع Churn
    for column in categorical_columns:

        # Crosstab كتحسب النسب حسب كل catégorie
        cross_table = pd.crosstab(
            df[column],
            df[target],
            normalize="index"
        ) * 100

        print(
            f"\n===== {column} vs {target} ====="
        )

        print(
            cross_table.round(2)
        )

        # كنرسمو النسب
        cross_table.plot(
            kind="bar",
            stacked=True,
            figsize=(10, 5)
        )

        plt.title(
            f"{target} selon {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Pourcentage")

        plt.xticks(rotation=45)

        plt.legend(
            title=target
        )

        plt.tight_layout()
        plt.show()
