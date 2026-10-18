
"""Pruebas unitarias de la configuración experimental."""

from pathlib import Path

import pytest

from clouds_cnn.config import ExperimentConfig
from clouds_cnn.utils import PROJECT_ROOT, VALID_DEVICE_PREFERENCES


def test_default_data_root_points_to_processed() -> None:
    """Comprueba que los datos de entrenamiento apunten a data/processed."""
    config = ExperimentConfig()

    assert config.data_root == PROJECT_ROOT / "data" / "processed"


def test_default_output_root_points_to_outputs() -> None:
    """Comprueba la ruta predeterminada de los resultados."""
    config = ExperimentConfig()

    assert config.output_root == PROJECT_ROOT / "outputs"


def test_default_experiment_parameters() -> None:
    """Comprueba los parámetros predeterminados principales."""
    config = ExperimentConfig()

    assert config.split_seed == 2026
    assert config.train_seed == 2026
    assert config.image_size == 256
    assert config.num_classes == 11
    assert config.batch_size == 32
    assert config.num_workers == 0
    assert config.epochs == 30


def test_paths_are_converted_to_path_objects(tmp_path: Path) -> None:
    """Acepta rutas de entrada como cadenas y las convierte a Path."""
    data_root = str(tmp_path / "processed")
    output_root = str(tmp_path / "outputs")

    config = ExperimentConfig(
        data_root=data_root,
        output_root=output_root,
    )

    assert isinstance(config.data_root, Path)
    assert isinstance(config.output_root, Path)
    assert config.data_root == Path(data_root)
    assert config.output_root == Path(output_root)


def test_accepts_valid_device_preferences() -> None:
    """Acepta todas las preferencias de dispositivo admitidas."""
    for preference in VALID_DEVICE_PREFERENCES:
        config = ExperimentConfig(device_preference=preference)

        assert config.device_preference == preference


def test_rejects_invalid_device_preference() -> None:
    """Rechaza preferencias de dispositivo no reconocidas."""
    with pytest.raises(ValueError):
        ExperimentConfig(device_preference="quantum")
