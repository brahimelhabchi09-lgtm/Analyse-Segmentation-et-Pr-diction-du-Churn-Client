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
