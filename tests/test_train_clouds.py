"""Pruebas del CLI adaptado del profesor a once clases CCSN.

Requiere MLflow, igual que el entry point real de entrenamiento.
"""

import pytest

pytest.importorskip("mlflow")

from clouds_cnn.config import ExperimentConfig
from clouds_cnn.train import build_parser, parse_config


def test_train_defaults_keep_eleven_classes_and_256_pixels():
    """Los flags omitidos heredan los valores de config.py."""
    assert vars(build_parser().parse_args([])) == {}
    config = parse_config([])
    assert config == ExperimentConfig()
    assert (config.num_classes, config.image_size) == (11, 256)


def test_train_flags_control_architecture_and_training():
    """Los flags CLI modifican parámetros sin cambiar clases ni resolución."""
    config = parse_config([
        "--run-name", "batchnorm_dropout",
        "--use-batchnorm",
        "--dropout-p", "0.3",
        "--epochs", "2",
        "--batch-size", "4",
        "--device", "cpu",
    ])
    assert config.use_batchnorm is True
    assert config.dropout_p == 0.3
    assert config.epochs == 2 and config.batch_size == 4
    assert config.num_classes == 11 and config.image_size == 256
