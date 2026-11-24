import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

DATA_PATH = "adult.data"
MODEL_PATH = "kmeans_model.joblib"
REPO_URL = "https://github.com/SeifAL111/mlops-prefect-kmeans.git"
REPO_BRANCH = "master"


@task(name="0-fetch-project")
def task_fetch_project(
    repo_url: str = REPO_URL,
    target_dir: str = ".",
    branch: str = REPO_BRANCH,
) -> None:
    """Récupère ou met à jour le projet depuis le dépôt Git distant."""
    logger = get_run_logger()
    target = Path(target_dir)

    if (target / ".git").exists():
        subprocess.run(
            ["git", "-C", str(target), "pull", "origin", branch],
            check=True,
        )
        logger.info("Dépôt mis à jour (git pull origin %s) dans %s.", branch, target)
    else:
        subprocess.run(
            ["git", "clone", "-b", branch, repo_url, str(target)],
            check=True,
        )
        logger.info("Dépôt cloné depuis %s dans %s.", repo_url, target)


@task(name="1-install-dependencies")
def task_install_dependencies(requirements_file: str = "requirements.txt") -> None:
    """Installe les dépendances depuis requirements.txt."""
    logger = get_run_logger()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements_file],
        check=True,
    )
    logger.info("Dépendances vérifiées/installées depuis %s.", requirements_file)


@task(name="2-prepare-data")
def task_prepare_data(file_path: str = DATA_PATH):
    """Prépare et normalise les données."""
    return prepare_data(file_path)


@task(name="3-train-model")
def task_train_model(X_processed, n_clusters: int = 4):
    """Entraîne le modèle K-Means."""
    return train_model(X_processed, n_clusters=n_clusters)


@task(name="4-save-model")
def task_save_model(model, filepath: str = MODEL_PATH) -> str:
    """Sauvegarde le modèle au format joblib."""
    save_model(model, filepath)
    return filepath


@task(name="5-load-model")
def task_load_model(filepath: str = MODEL_PATH):
    """Charge un modèle sauvegardé."""
    return load_model(filepath)


@task(name="6-evaluate-model")
def task_evaluate_model(model, X_processed, sample_size: int = 5000) -> float:
    """Évalue le modèle avec le score de silhouette."""
    return evaluate_model(model, X_processed[:sample_size])


@flow(name="all", log_prints=True)
def flow_all(
    file_path: str = DATA_PATH,
    n_clusters: int = 4,
    model_path: str = MODEL_PATH,
    fetch_repo: bool = False,
):
    """Exécute la récupération optionnelle puis les étapes 2, 3, 4 et 6."""
    if fetch_repo:
        task_fetch_project()
    X_scaled, _ = task_prepare_data(file_path)
    model = task_train_model(X_scaled, n_clusters=n_clusters)
    task_save_model(model, model_path)
    task_evaluate_model(model, X_scaled)


@flow(name="train", log_prints=True)
def flow_train(file_path: str = DATA_PATH, n_clusters: int = 4):
    """Prépare les données et entraîne le modèle."""
    X_scaled, _ = task_prepare_data(file_path)
    return task_train_model(X_scaled, n_clusters=n_clusters)


@flow(name="evaluate", log_prints=True)
def flow_evaluate(file_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    """Charge le modèle et l'évalue."""
    X_scaled, _ = task_prepare_data(file_path)
    model = task_load_model(model_path)
    return task_evaluate_model(model, X_scaled)


FLOWS = {
    "all": flow_all,
    "train": flow_train,
    "evaluate": flow_evaluate,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect MLOps")
    parser.add_argument(
        "--flow",
        choices=FLOWS.keys(),
        default="all",
        help="Nom du flow à exécuter (all, train, evaluate)",
    )
    args = parser.parse_args()
    FLOWS[args.flow]()
