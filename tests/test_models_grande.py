"""Pruebas de la CNN de escala grande."""

import pytest
import torch
from torch import nn

from clouds_cnn.models_grande import (
    DILATIONS,
    KERNELS,
    POOL_TYPES,
    CloudCNNGrande,
)


@pytest.fixture
def model():
    """Instancia ligera para las pruebas."""
    return CloudCNNGrande(num_classes=11, dropout_p=0.3)


def test_forma_logits(model):
    model.eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 11)
    assert torch.isfinite(logits).all()


def test_bloques_y_escalas(model):
    assert len(model.features) == 4
    for i, bloque in enumerate(model.features):
        assert bloque.conv.kernel_size == (KERNELS[i],) * 2
        assert bloque.conv.dilation == (DILATIONS[i],) * 2
        expected = nn.MaxPool2d if POOL_TYPES[i] == "max" else nn.AvgPool2d
        assert isinstance(bloque.pool, expected)
        assert isinstance(bloque.act, nn.LeakyReLU)


def test_batchnorm_opcional():
    model = CloudCNNGrande(use_batchnorm=False)
    assert all(isinstance(b.norm, nn.Identity) for b in model.features)


def test_dropout_configurable():
    assert isinstance(CloudCNNGrande(dropout_p=0).classifier.dropout, nn.Identity)
    assert isinstance(CloudCNNGrande(dropout_p=0.3).classifier.dropout, nn.Dropout)


@pytest.mark.parametrize("classes", [0, 1])
def test_clases_invalidas(classes):
    with pytest.raises(ValueError, match="num_classes"):
        CloudCNNGrande(num_classes=classes)


@pytest.mark.parametrize("p", [-0.1, 1.0])
def test_dropout_invalido(p):
    with pytest.raises(ValueError, match="dropout_p"):
        CloudCNNGrande(dropout_p=p)


def test_backward(model):
    model.train()
    loss = model(torch.randn(2, 3, 32, 32)).square().mean()
    loss.backward()
    assert model.features.block1.conv.weight.grad is not None


def test_no_softmax(model):
    assert not any(isinstance(layer, nn.Softmax) for layer in model.modules())
