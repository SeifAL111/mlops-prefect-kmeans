from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def main():
    print("--- Début du pipeline MLOps ---")

    # 1. Préparation des données
    X_scaled, _scaler = prepare_data("adult.data")

    # 2. Entraînement du modèle de clustering
    model = train_model(X_scaled, n_clusters=4)

    # 3. Évaluation du modèle (Score de silhouette)
    # (On prend un échantillon de 5000 lignes pour accélérer le calcul de la silhouette sur ce grand dataset)
    evaluate_model(model, X_scaled[:5000])

    # 4. Sauvegarde du modèle
    save_model(model, "kmeans_model.joblib")

    # 5. Test du chargement du modèle
    load_model("kmeans_model.joblib")

    print("--- Pipeline exécuté avec succès ! ---")


if __name__ == "__main__":
    main()
