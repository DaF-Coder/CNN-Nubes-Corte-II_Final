"""Integración mínima de orquestación y MLflow para once clases CCSN.

La suite omite este archivo cuando MLflow no está instalado.
"""


import pytest

pytest.importorskip("mlflow")

from clouds_cnn.config import ExperimentConfig
from clouds_cnn.experiment import build_model, resolve_tracking_uri


def test_experiment_builds_eleven_class_model():
    """La orquestación construye la arquitectura y salida correctas."""
    config = ExperimentConfig()
    model = build_model(config)
    assert model.classifier.fc2.out_features == 11


def test_tracking_uri_is_defined():
    """El backend de MLflow debe resolverse antes del entrenamiento."""
    assert resolve_tracking_uri()
