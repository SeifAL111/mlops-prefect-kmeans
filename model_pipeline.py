import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def prepare_data(file_path="adult.data"):
    """
    Charge et prétraite le dataset Adult pour le clustering K-Means.
    """
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    # Chargement du fichier
    df = pd.read_csv(file_path, names=col_names, sep=r"\s*,\s*", engine="python")

    # Sélection des variables numériques clés pour le K-Means
    features = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    X = df[features].copy()

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Données préparées et normalisées avec succès pour le K-Means !")
    return X_scaled, scaler


def train_model(X_processed, n_clusters=4):
    """Entraîne un modèle K-Means sur les données normalisées."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    print(f"Modèle K-Means entraîné avec succès (k={n_clusters}).")
    return kmeans


def save_model(model, filepath="kmeans_model.joblib"):
    """Sauvegarde le modèle sur le disque."""
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé dans : {filepath}")


def evaluate_model(model, X_processed):
    """Évalue le modèle de clustering (score de silhouette)."""
    # Prédiction des labels pour l'échantillon fourni
    labels = model.predict(X_processed)
    score = silhouette_score(X_processed, labels)
    print(f"Score de silhouette du clustering : {score:.4f}")
    return score


def load_model(filepath="kmeans_model.joblib"):
    """Charge un modèle sauvegardé depuis le disque."""
    model = joblib.load(filepath)
    print(f"Modèle chargé avec succès depuis : {filepath}")
    return model
