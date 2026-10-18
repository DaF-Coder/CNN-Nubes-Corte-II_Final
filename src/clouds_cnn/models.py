# AJUSTES POR ETAPA (nubes CCSN):
# Editable: FEATURE_CHANNELS, POOLED_SIZE, HIDDEN_UNITS y los bloques de CloudCNN.
# La capa fc2 debe producir config.num_classes=11 logits.


"""Arquitectura CNN para clasificar imágenes de nubes.

El módulo define exclusivamente el CNN y el conteo de parámetros.
No gestiona optimizadores, DataLoaders ni ciclos de entrenamiento.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

FEATURE_CHANNELS: tuple[int, ...] = (32, 64, 128, 256)
POOLED_SIZE: tuple[int, int] = (2, 2)
HIDDEN_UNITS: int = 128
INPUT_CHANNELS: int = 3


def _conv_block(
    in_channels: int,
    out_channels: int,
    use_batchnorm: bool,
) -> nn.Sequential:
    """Construye un bloque convolucional con pooling.

    Args:
        in_channels: Número de canales de entrada.
        out_channels: Número de canales producidos por la convolución.
        use_batchnorm: Indica si se utiliza Batch Normalization.

    Returns:
        Bloque Conv2d -> BatchNorm2d/Identity -> ReLU -> MaxPool2d.
    """
    return nn.Sequential(
        OrderedDict(
            conv=nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            ),
            norm=(
                nn.BatchNorm2d(out_channels)
                if use_batchnorm
                else nn.Identity()
            ),
            act=nn.ReLU(),
            pool=nn.MaxPool2d(kernel_size=2, stride=2),
        )
    )


class CloudCNN(nn.Module):
    """CNN de cuatro bloques para clasificar imágenes RGB de nubes.

    La red utiliza cuatro bloques convolucionales, pooling adaptativo
    y un clasificador totalmente conectado.

    Args:
        num_classes: Número de clases de salida; debe ser al menos 2.
        use_batchnorm: Activa Batch Normalization en los bloques.
        dropout_p: Probabilidad de Dropout, en el intervalo [0, 1).

    Raises:
        ValueError: Si el número de clases o el Dropout son inválidos.
    """

    def __init__(
        self,
        num_classes: int,
        use_batchnorm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                f"num_classes debe ser >= 2; se recibió {num_classes}"
            )
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(
                f"dropout_p debe estar en [0, 1); se recibió {dropout_p}"
            )

        in_channels = (INPUT_CHANNELS, *FEATURE_CHANNELS[:-1])
        blocks = (
            (f"block{i}", _conv_block(c_in, c_out, use_batchnorm))
            for i, (c_in, c_out) in enumerate(
                zip(in_channels, FEATURE_CHANNELS, strict=True),
                start=1,
            )
        )
        self.features = nn.Sequential(OrderedDict(blocks))
        self.pool = nn.AdaptiveAvgPool2d(POOLED_SIZE)

        flat_features = (
            FEATURE_CHANNELS[-1] * POOLED_SIZE[0] * POOLED_SIZE[1]
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                flatten=nn.Flatten(),
                fc1=nn.Linear(flat_features, HIDDEN_UNITS),
                act=nn.ReLU(),
                dropout=(
                    nn.Dropout(dropout_p)
                    if dropout_p > 0.0
                    else nn.Identity()
                ),
                fc2=nn.Linear(HIDDEN_UNITS, num_classes),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula los logits para un lote de imágenes RGB.

        Args:
            x: Tensor con forma (batch_size, 3, height, width).
                Las dimensiones espaciales deben permitir cuatro pooling.

        Returns:
            Tensor de logits con forma (batch_size, num_classes).
        """
        features = self.features(x)
        pooled = self.pool(features)
        return self.classifier(pooled)


def count_parameters(model: nn.Module) -> int:
    """Cuenta todos los parámetros, entrenables y no entrenables."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Cuenta los parámetros cuyo gradiente está habilitado."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
