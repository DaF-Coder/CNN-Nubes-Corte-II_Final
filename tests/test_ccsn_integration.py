"""Integración CCSN: 11 etiquetas, Resize 256 y CNN del profesor adaptada.

Las imágenes son sintéticas: nunca se requieren las fotografías reales en CI.
Los tests matemáticos con menos clases permanecen en otros módulos.
"""

import json

import numpy as np
import torch
from PIL import Image
from torch import nn

from clouds_cnn.config import ExperimentConfig
from clouds_cnn.data import DATASET_STATS_NAME, make_dataloaders
from clouds_cnn.models import CloudCNN

CLASSES = ("Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St")


def test_ccsn_config_defaults():
    """Confirma el contrato de clasificación de nubes a 256 píxeles."""
    config = ExperimentConfig()
    assert config.num_classes == 11
    assert config.image_size == 256


def test_ccsn_model_accepts_256_rgb_images():
    """Comprueba el tamaño de salida y la retropropagación con 11 logits."""
    model = CloudCNN(num_classes=11)
    x = torch.randn(1, 3, 256, 256)
    y = torch.tensor([10])
    logits = model(x)
    assert logits.shape == (1, 11)
    nn.CrossEntropyLoss()(logits, y).backward()
    assert model.classifier.fc2.weight.grad is not None


def test_ccsn_dataloaders_and_model_connect(tmp_path):
    """Conecta ImageFolder, transformaciones, loaders y forward de la CNN.

    Args:
        tmp_path: Directorio temporal para las imágenes sintéticas.
    """
    root = tmp_path / "processed"
    rng = np.random.default_rng(2026)
    for split in ("train", "val", "test"):
        for name in CLASSES:
            folder = root / split / name
            folder.mkdir(parents=True)
            Image.fromarray(rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)).save(folder / "0.png")
    stats = {
        "split_seed": 2026,
        "class_to_label": {name: i for i, name in enumerate(CLASSES)},
        "transform": {"image_size": 256, "center_crop": None},
        "mean_train": [0.5] * 3,
        "std_train": [0.25] * 3,
    }
    (root / DATASET_STATS_NAME).write_text(json.dumps(stats), encoding="utf-8")
    config = ExperimentConfig(data_root=root, batch_size=2, device_preference="cpu")
    loaders = make_dataloaders(config, torch.device("cpu"))
    images, labels = next(iter(loaders.train))
    assert loaders.class_names == list(CLASSES)
    assert images.shape == (2, 3, 256, 256)
    assert labels.min() >= 0 and labels.max() < 11
    logits = CloudCNN(num_classes=config.num_classes)(images)
    assert logits.shape == (2, 11)
