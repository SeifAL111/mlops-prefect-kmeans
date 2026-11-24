import numpy as np
from sklearn.cluster import KMeans

from model_pipeline import load_model, prepare_data, save_model, train_model


def test_prepare_and_train_kmeans():
    X_scaled, _ = prepare_data("adult.data")
    assert X_scaled.shape[0] > 0
    assert not np.isnan(X_scaled).any()

    model = train_model(X_scaled, n_clusters=4)
    assert isinstance(model, KMeans)
    assert model.n_clusters == 4


def test_save_and_load_joblib(tmp_path):
    X_scaled, _ = prepare_data("adult.data")
    model = train_model(X_scaled, n_clusters=4)

    model_file = tmp_path / "test_kmeans.joblib"
    save_model(model, str(model_file))
    assert model_file.exists()

    loaded = load_model(str(model_file))
    assert isinstance(loaded, KMeans)
    assert loaded.n_clusters == 4
