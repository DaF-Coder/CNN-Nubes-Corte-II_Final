
"""Pruebas unitarias de la arquitectura CNN para nubes."""

import pytest
import torch
from torch import nn

from clouds_cnn.models import (
    CloudCNN,
    count_parameters,
    count_trainable_parameters,
)


@pytest.mark.parametrize("num_classes", [2, 4, 5])
def test_forward_returns_expected_shape(num_classes: int) -> None:
    """La salida debe tener una columna por clase."""
    model = CloudCNN(num_classes=num_classes)
    images = torch.randn(2, 3, 64, 64)

    logits = model(images)

    assert logits.shape == (2, num_classes)


def test_forward_supports_backward_pass() -> None:
    """La pérdida debe poder propagarse hacia los parámetros."""
    model = CloudCNN(num_classes=4)
    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 3])
    loss_fn = nn.CrossEntropyLoss()

    loss = loss_fn(model(images), labels)
    loss.backward()

    assert loss.item() > 0
    assert model.classifier.fc2.weight.grad is not None


def test_batchnorm_enabled() -> None:
    """Cada bloque debe usar BatchNorm2d cuando se activa."""
    model = CloudCNN(num_classes=4, use_batchnorm=True)

    for index in range(1, 5):
        block = getattr(model.features, f"block{index}")

        assert isinstance(block.norm, nn.BatchNorm2d)
        assert block.conv.bias is None


def test_batchnorm_disabled() -> None:
    """Sin BatchNorm, los bloques usan Identity y bias convolucional."""
    model = CloudCNN(num_classes=4, use_batchnorm=False)

    for index in range(1, 5):
        block = getattr(model.features, f"block{index}")

        assert isinstance(block.norm, nn.Identity)
        assert block.conv.bias is not None


@pytest.mark.parametrize(
    ("dropout_p", "expected_type"),
    [
        (0.0, nn.Identity),
        (0.25, nn.Dropout),
        (0.5, nn.Dropout),
    ],
)
def test_dropout_configuration(
    dropout_p: float,
    expected_type: type[nn.Module],
) -> None:
    """El clasificador debe respetar la configuración de Dropout."""
    model = CloudCNN(num_classes=4, dropout_p=dropout_p)

    assert isinstance(model.classifier.dropout, expected_type)

    if dropout_p > 0:
        assert model.classifier.dropout.p == dropout_p


@pytest.mark.parametrize("num_classes", [0, 1, -2])
def test_invalid_num_classes_raises_error(num_classes: int) -> None:
    """El modelo no debe aceptar menos de dos clases."""
    with pytest.raises(ValueError, match="num_classes"):
        CloudCNN(num_classes=num_classes)


@pytest.mark.parametrize("dropout_p", [-0.1, 1.0, 1.5])
def test_invalid_dropout_raises_error(dropout_p: float) -> None:
    """La probabilidad de Dropout debe pertenecer a [0, 1)."""
    with pytest.raises(ValueError, match="dropout_p"):
        CloudCNN(num_classes=4, dropout_p=dropout_p)


def test_parameter_counts_are_consistent() -> None:
    """Los parámetros entrenables no deben superar el total."""
    model = CloudCNN(num_classes=4)

    total = count_parameters(model)
    trainable = count_trainable_parameters(model)

    assert total > 0
    assert trainable == total


def test_frozen_parameters_are_not_counted_as_trainable() -> None:
    """Los parámetros congelados solo cuentan en el total."""
    model = CloudCNN(num_classes=4)
    model.classifier.fc2.weight.requires_grad = False

    total = count_parameters(model)
    trainable = count_trainable_parameters(model)

    assert total > trainable
    assert total - trainable == model.classifier.fc2.weight.numel()


def test_adaptive_pool_accepts_different_image_sizes() -> None:
    """El pooling adaptativo permite variar el tamaño de las imágenes."""
    model = CloudCNN(num_classes=4).eval()

    with torch.no_grad():
        small_logits = model(torch.randn(1, 3, 32, 32))
        large_logits = model(torch.randn(1, 3, 96, 96))

    assert small_logits.shape == (1, 4)
    assert large_logits.shape == (1, 4)
