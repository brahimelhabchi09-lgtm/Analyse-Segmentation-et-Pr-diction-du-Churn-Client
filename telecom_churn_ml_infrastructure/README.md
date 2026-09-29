# Telecom Churn ML - Infrastructure Docker

Infrastructure complète pour un projet Machine Learning de segmentation
clients et prédiction du churn.

## Services

- Streamlit : http://localhost:8501
- MLflow : http://localhost:5000
- PostgreSQL : localhost:5432

## Architecture

Streamlit
   |
   +----> Models
   |
   +----> MLflow :5000
              |
              +----> PostgreSQL :5432

## Lancer le projet

Depuis le dossier racine :

```bash
docker compose up --build
```

Puis ouvrir :

- http://localhost:8501
- http://localhost:5000

## Arrêter

```bash
docker compose down
```

## Arrêter et supprimer les volumes

Attention : cette commande supprime les données PostgreSQL et MLflow stockées
dans les volumes Docker.

```bash
docker compose down -v
```

## Structure

```text
telecom_churn_ml/
├── app/
├── artifacts/
├── config/
├── data/
├── mlruns/
├── models/
├── notebooks/
├── src/
├── .dockerignore
├── .env
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
```

## Ajouter le dataset

Placez votre dataset ici :

```text
data/raw/telecom_churn.csv
```

La colonne cible doit être :

```text
churn
```

## Important

Le service Streamlit utilise :

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
```

Dans Docker, il faut utiliser `mlflow:5000` et non `localhost:5000`
pour communiquer entre les conteneurs.
