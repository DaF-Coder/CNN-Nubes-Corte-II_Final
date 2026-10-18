
"""Pruebas unitarias de carga y validación de datos."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from clouds_cnn.data import (
    DATASET_STATS_NAME,
    SPLITS,
    load_dataset_stats,
    make_dataloaders,
    make_split_loader,
)


def test_make_dataloaders_builds_all_splits(config) -> None:
    """Comprueba que se construyan los tres DataLoaders."""
    loaders = make_dataloaders(config, device=torch.device("cpu"))

    assert loaders.train is not None
    assert loaders.val is not None
    assert loaders.test is not None
    assert set(SPLITS) == {"train", "val", "test"}


def test_dataloaders_use_expected_class_names(config) -> None:
    """Comprueba que las clases coincidan con las carpetas procesadas."""
    loaders = make_dataloaders(config, device=torch.device("cpu"))

    assert loaders.class_names == ["Ac", "As", "Cb"]
    assert loaders.train.dataset.classes == loaders.class_names
    assert loaders.val.dataset.classes == loaders.class_names
    assert loaders.test.dataset.classes == loaders.class_names


def test_train_batch_has_expected_shape_and_types(config) -> None:
    """Comprueba las dimensiones y los tipos de un lote de entrenamiento."""
    loaders = make_dataloaders(config, device=torch.device("cpu"))
    images, labels = next(iter(loaders.train))

    assert images.ndim == 4
    assert images.shape[1:] == (3, config.image_size, config.image_size)
    assert images.dtype == torch.float32
    assert labels.ndim == 1
    assert labels.dtype == torch.int64


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_labels_are_within_valid_class_range(config, split: str) -> None:
    """Comprueba que las etiquetas sean índices de clases válidos."""
    loader = make_split_loader(
        config,
        split,
        device=torch.device("cpu"),
    )
    _, labels = next(iter(loader))

    assert torch.all(labels >= 0)
    assert torch.all(labels < config.num_classes)


def test_invalid_split_raises_value_error(config) -> None:
    """Rechaza nombres de partición que no están permitidos."""
    with pytest.raises(ValueError, match="split"):
        make_split_loader(
            config,
            "development",
            device=torch.device("cpu"),
        )


def test_missing_dataset_stats_raises_file_not_found(
    config,
    tmp_path: Path,
) -> None:
    """Informa cuando falta el archivo de estadísticas."""
    missing_root = tmp_path / "missing"
    invalid_config = replace(config, data_root=missing_root)

    with pytest.raises(FileNotFoundError):
        load_dataset_stats(invalid_config.data_root)


def test_mismatched_split_seed_raises_value_error(config) -> None:
    """Rechaza una semilla distinta de la usada para preparar los datos."""
    invalid_config = replace(config, split_seed=config.split_seed + 1)

    with pytest.raises(ValueError, match="split_seed"):
        make_split_loader(
            invalid_config,
            "train",
            device=torch.device("cpu"),
        )


def test_mismatched_image_size_raises_value_error(config) -> None:
    """Rechaza una resolución distinta de la registrada en las estadísticas."""
    invalid_config = replace(config, image_size=config.image_size + 16)

    with pytest.raises(ValueError, match="image_size"):
        make_split_loader(
            invalid_config,
            "train",
            device=torch.device("cpu"),
        )


def test_wrong_num_classes_raises_value_error(config) -> None:
    """Rechaza una cantidad de clases incompatible con el dataset."""
    invalid_config = replace(config, num_classes=config.num_classes + 1)

    with pytest.raises(ValueError, match="num_classes"):
        make_split_loader(
            invalid_config,
            "train",
            device=torch.device("cpu"),
        )


def test_dataset_stats_file_exists(prepared_dataset: Path) -> None:
    """Comprueba que la preparación genere las estadísticas."""
    stats_path = prepared_dataset / DATASET_STATS_NAME

    assert stats_path.is_file()


def test_dataset_stats_contain_valid_channel_statistics(
    prepared_dataset: Path,
) -> None:
    """Comprueba que las estadísticas RGB tengan tres valores válidos."""
    stats = load_dataset_stats(prepared_dataset)

    assert len(stats["mean_train"]) == 3
    assert len(stats["std_train"]) == 3
    assert all(value > 0 for value in stats["std_train"])
