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
