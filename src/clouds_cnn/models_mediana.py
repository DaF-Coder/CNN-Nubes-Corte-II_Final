"""CNN de escala mediana para imágenes RGB de nubes."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

CHANNELS = (32, 64, 128, 256)
KERNELS = (5, 3, 3, 3)
DILATIONS = (1, 1, 1, 1)
POOL_TYPES = ('max', 'avg', 'max', 'avg')


def _block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int,
    pooling: str,
    use_batchnorm: bool,
) -> nn.Sequential:
    """Construye un bloque convolucional con reducción espacial."""
    pool = nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2)
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation, bias=not use_batchnorm,
        )),
        ("norm", nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()),
        ("act", nn.LeakyReLU(negative_slope=0.1)),
        ("pool", pool),
    ]))


class CloudCNNMediana(nn.Module):
    """Extrae características de escala mediana y produce logits."""

    def __init__(
        self,
        num_classes: int = 11,
        use_batchnorm: bool = True,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes debe ser al menos 2")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p debe estar en [0, 1)")

        entradas = (3, *CHANNELS[:-1])
        bloques = [
            (f"block{i}", _block(c_in, c_out, kernel, dilation, pool, use_batchnorm))
            for i, (c_in, c_out, kernel, dilation, pool) in enumerate(
                zip(entradas, CHANNELS, KERNELS, DILATIONS, POOL_TYPES, strict=True),
                start=1,
            )
        ]
        self.features = nn.Sequential(OrderedDict(bloques))
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(CHANNELS[-1] * 4, 128)),
            ("act", nn.LeakyReLU(negative_slope=0.1)),
            ("dropout", nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()),
            ("fc2", nn.Linear(128, num_classes)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve logits con forma (lote, num_classes)."""
        return self.classifier(self.pool(self.features(x)))
