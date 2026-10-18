
"""Fixtures compartidas para las pruebas de clouds_cnn."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from clouds_cnn.config import ExperimentConfig
from clouds_cnn.preparation import prepare

CLASS_NAMES = ("Ac", "As", "Cb")
SAMPLES_PER_CLASS = 12
IMAGE_SIZE = 16
SPLIT_SEED = 2026


@pytest.fixture
def source_dataset(tmp_path: Path) -> Path:
    """Crea imágenes RGB sintéticas organizadas por clase.

    Args:
        tmp_path: Directorio temporal proporcionado por pytest.

    Returns:
        Ruta a la carpeta con las imágenes originales de prueba.
    """
    source = tmp_path / "curated"
    source.mkdir()

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = source / class_name
        class_dir.mkdir()

        for image_index in range(SAMPLES_PER_CLASS):
            color = (
                (class_index * 47 + image_index * 17 + 11) % 256,
                (class_index * 31 + image_index * 23 + 29) % 256,
                (class_index * 61 + image_index * 13 + 43) % 256,
            )
            image = Image.new("RGB", (24, 24), color)
            image.save(class_dir / f"image_{image_index:03d}.png")

    return source


@pytest.fixture
def prepared_dataset(source_dataset: Path, tmp_path: Path) -> Path:
    """Prepara los datos temporales en train, val y test.

    Args:
        source_dataset: Imágenes originales creadas por otra fixture.
        tmp_path: Directorio temporal de pytest.

    Returns:
        Ruta a la carpeta procesada con splits y estadísticas.
    """
    output = tmp_path / "processed"
    prepare(
        source=source_dataset,
        output=output,
        split_seed=SPLIT_SEED,
        image_size=IMAGE_SIZE,
        num_workers=0,
    )
    return output


@pytest.fixture
def config(prepared_dataset: Path) -> ExperimentConfig:
    """Construye una configuración compatible con el dataset temporal.

    Args:
        prepared_dataset: Datos procesados para los tests.

    Returns:
        Configuración pequeña para ejecutar pruebas en CPU.
    """
    return ExperimentConfig(
        data_root=prepared_dataset,
        num_classes=len(CLASS_NAMES),
        split_seed=SPLIT_SEED,
        image_size=IMAGE_SIZE,
        batch_size=4,
        num_workers=0,
        device_preference="cpu",
    )


@pytest.fixture
def cpu_device() -> torch.device:
    """Devuelve el dispositivo CPU para pruebas reproducibles."""
    return torch.device("cpu")


@pytest.fixture(scope="session", autouse=True)
def isolated_mlflow_backend(tmp_path_factory):
    """Aísla MLflow de la base local del proyecto durante los tests.

    Args:
        tmp_path_factory: Creador de directorios temporales de pytest.

    Yields:
        Path: Base SQLite temporal del backend de seguimiento.
    """
    database = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    patcher = pytest.MonkeyPatch()
    patcher.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{database.as_posix()}")
    yield database
    patcher.undo()
