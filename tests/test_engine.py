"""Pruebas unitarias del motor de entrenamiento."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from clouds_cnn.config import ExperimentConfig
from clouds_cnn.engine import (
    EpochMetrics,
    evaluate,
    fit,
    save_checkpoint,
    train_one_epoch,
)


@pytest.fixture
def device() -> torch.device:
    """Proporciona el dispositivo CPU para pruebas reproducibles."""
    return torch.device("cpu")


@pytest.fixture
def loader() -> DataLoader:
    """Construye un loader pequeño para clasificación binaria."""
    X = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [-1.0, -1.0],
            [0.5, -0.5],
            [-0.5, 0.5],
        ]
    )
    y = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1])
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def model() -> nn.Module:
    """Construye un clasificador lineal de dos clases."""
    torch.manual_seed(42)
    return nn.Linear(2, 2)


@pytest.fixture
def loss_fn() -> nn.Module:
    """Proporciona la pérdida estándar para clasificación."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def config() -> ExperimentConfig:
    """Construye una configuración con dos épocas."""
    return ExperimentConfig(epochs=2)


def test_train_one_epoch_returns_metrics(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> None:
    """Entrenar una época devuelve métricas válidas."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = train_one_epoch(model, loader, loss_fn, optimizer, device)

    assert isinstance(metrics, EpochMetrics)
    assert metrics.loss > 0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert model.training


def test_train_one_epoch_updates_parameters(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> None:
    """Los parámetros cambian durante el entrenamiento."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    train_one_epoch(model, loader, loss_fn, optimizer, device)

    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )


def test_evaluate_does_not_update_parameters(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> None:
    """Evaluar no modifica los parámetros y deja el modelo en eval."""
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = evaluate(model, loader, loss_fn, device)

    assert metrics.loss > 0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert not model.training
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))


def test_evaluate_calls_on_batch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> None:
    """El callback recibe logits y etiquetas en cada lote."""
    received: list[tuple[torch.Size, torch.Size]] = []

    def callback(logits: torch.Tensor, labels: torch.Tensor) -> None:
        received.append((logits.shape, labels.shape))

    evaluate(model, loader, loss_fn, device, on_batch=callback)

    assert len(received) == len(loader)
    assert all(logits == (4, 2) for logits, _ in received)
    assert all(labels == (4,) for _, labels in received)


def test_empty_loader_raises_value_error(
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> None:
    """Un loader vacío produce un error explícito."""
    dataset = TensorDataset(
        torch.empty((0, 2)),
        torch.empty((0,), dtype=torch.long),
    )
    empty_loader = DataLoader(dataset, batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="ninguna muestra"):
        train_one_epoch(model, empty_loader, loss_fn, optimizer, device)


def test_non_mean_reduction_raises_value_error(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """El motor rechaza una pérdida que suma en lugar de promediar."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="reduction='mean'"):
        train_one_epoch(model, loader, loss_fn, optimizer, device)


def test_fit_saves_best_checkpoint(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    """Fit registra las épocas y guarda el checkpoint de menor val_loss."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"

    history = fit(
        model,
        loader,
        loader,
        loss_fn,
        optimizer,
        device,
        config,
        checkpoint_path,
        verbose=False,
    )

    assert len(history["train_loss"]) == config.epochs
    assert len(history["train_acc"]) == config.epochs
    assert len(history["val_loss"]) == config.epochs
    assert len(history["val_acc"]) == config.epochs
    assert len(history["lr"]) == config.epochs
    assert len(history["epoch_time"]) == config.epochs
    assert history["best_epoch"] in range(1, config.epochs + 1)
    assert history["best_val_loss"] == min(history["val_loss"])
    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["epoch"] == history["best_epoch"]
    assert checkpoint["val_loss"] == history["best_val_loss"]
    assert checkpoint["config"] == config.to_dict()


def test_save_checkpoint_creates_parent_directory(
    model: nn.Module,
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    """Guardar un checkpoint crea la carpeta de destino."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "nested" / "best.pt"

    save_checkpoint(path, 1, model, optimizer, 0.5, config)

    assert path.exists()



def test_fit_rejects_non_finite_validation_loss(
    loader: DataLoader,
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Fit falla si ninguna época obtiene una pérdida finita."""
    config = ExperimentConfig(epochs=1)

    class NonFiniteModel(nn.Module):
        """Modelo de prueba que produce logits no finitos."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            logits = self.weight.unsqueeze(0).expand(X.shape[0], -1)
            return logits * float("nan")

    model = NonFiniteModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="no se guardó checkpoint"):
        fit(
            model,
            loader,
            loader,
            nn.CrossEntropyLoss(),
            optimizer,
            device,
            config,
            tmp_path / "best.pt",
            verbose=False,
        )
