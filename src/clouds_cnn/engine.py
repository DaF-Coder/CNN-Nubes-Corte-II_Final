# AJUSTES POR ETAPA (nubes CCSN):
# Editable: protocolo de entrenamiento (fit, train_one_epoch, evaluate).
# No evaluar test durante la selección del modelo.

"""Motor de entrenamiento: train_one_epoch, evaluate y fit.

El motor solo asume el contrato externo del modelo:
logits = model(X_batch)
X_batch: (B, C, H, W) -> logits: (B, num_classes).

No accede a submódulos internos, no conoce MLflow y nunca recibe el loader
de test. Entrena un número fijo de épocas y conserva el checkpoint con
menor val_loss.

Las métricas se acumulan como tensores en el dispositivo y se convierten
a float una sola vez por época.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

from clouds_cnn.config import ExperimentConfig

BatchCallback = Callable[[torch.Tensor, torch.Tensor], None]
"""Recibe (logits, y_batch) de cada batch, ya en el dispositivo."""


@dataclass(frozen=True)
class EpochMetrics:
    """Métricas de una pasada completa sobre un loader.

    Attributes:
        loss: Pérdida promedio por muestra.
        accuracy: Fracción de aciertos, en [0, 1].
    """

    loss: float
    accuracy: float


class History(TypedDict):
    """Historia del entrenamiento, con una entrada por época."""

    train_loss: list[float]
    train_acc: list[float]
    val_loss: list[float]
    val_acc: list[float]
    lr: list[float]
    epoch_time: list[float]
    best_epoch: int
    best_val_loss: float


def _check_mean_reduction(loss_fn: nn.Module) -> None:
    """Comprueba que la pérdida promedie sobre el batch."""
    reduction = getattr(loss_fn, "reduction", "mean")
    if reduction != "mean":
        raise ValueError(
            f"loss_fn debe promediar sobre el batch (reduction='mean'); tiene {reduction!r}"
        )


def _synchronize(device: torch.device) -> None:
    """Espera a que terminen los kernels CUDA para medir tiempos reales."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    on_batch: BatchCallback | None = None,
) -> EpochMetrics:
    """Recorre un loader y actualiza los pesos si recibe un optimizador."""
    _check_mean_reduction(loss_fn)
    non_blocking = device.type == "cuda"
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    num_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=non_blocking)
        y_batch = y_batch.to(device, non_blocking=non_blocking)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        if on_batch is not None:
            on_batch(logits.detach(), y_batch)

        batch_size = y_batch.size(0)
        loss_sum += loss.detach() * batch_size
        correct += (logits.argmax(dim=1) == y_batch).sum()
        num_samples += batch_size

    if num_samples == 0:
        raise ValueError("El loader no produjo ninguna muestra.")

    return EpochMetrics(
        loss=loss_sum.item() / num_samples,
        accuracy=correct.item() / num_samples,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """Entrena una época y actualiza los parámetros del modelo.

    Args:
        model: Modelo que produce logits con forma (B, num_classes).
        loader: Loader de imágenes y etiquetas.
        loss_fn: Pérdida con reduction='mean'.
        optimizer: Optimizador de los parámetros del modelo.
        device: Dispositivo en el que se ejecuta el modelo.

    Returns:
        Pérdida y accuracy promedio por muestra.

    Raises:
        ValueError: Si el loader está vacío o la reducción no es mean.
    """
    model.train()
    return _run_epoch(model, loader, loss_fn, device, optimizer)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    on_batch: BatchCallback | None = None,
) -> EpochMetrics:
    """Evalúa sin calcular gradientes ni actualizar los pesos.

    Args:
        model: Modelo que produce logits con forma (B, num_classes).
        loader: Loader del conjunto que se desea evaluar.
        loss_fn: Pérdida con reduction='mean'.
        device: Dispositivo en el que se ejecuta el modelo.
        on_batch: Callback opcional para recoger logits y etiquetas.

    Returns:
        Pérdida y accuracy promedio por muestra.

    Raises:
        ValueError: Si el loader está vacío o la reducción no es mean.
    """
    model.eval()
    with torch.inference_mode():
        return _run_epoch(
            model,
            loader,
            loss_fn,
            device,
            optimizer=None,
            on_batch=on_batch,
        )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    config: ExperimentConfig,
) -> None:
    """Guarda el modelo, el optimizador y la configuración del experimento."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": float(val_loss),
            "config": config.to_dict(),
        },
        path,
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: ExperimentConfig,
    checkpoint_path: Path,
    verbose: bool = True,
) -> History:
    """Entrena un número fijo de épocas y guarda el mejor checkpoint.

    La selección del checkpoint depende de la menor pérdida de validación.
    No utiliza el conjunto de test, scheduler ni early stopping.

    Args:
        model: Modelo que ya está en el dispositivo indicado.
        train_loader: Loader de entrenamiento.
        val_loader: Loader de validación.
        loss_fn: Pérdida con reduction='mean'.
        optimizer: Optimizador construido sobre los parámetros del modelo.
        device: Dispositivo de cómputo.
        config: Configuración; aporta epochs y to_dict().
        checkpoint_path: Ruta del mejor checkpoint.
        verbose: Si es True, imprime un resumen por época.

    Returns:
        Historial de métricas y datos de la mejor época.

    Raises:
        ValueError: Si un loader está vacío o la reducción no es mean.
        RuntimeError: Si ninguna época produce una val_loss finita.
    """
    history: History = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": [],
        "best_epoch": 0,
        "best_val_loss": math.inf,
    }

    for epoch in range(1, config.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        _synchronize(device)
        start = time.perf_counter()

        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        _synchronize(device)
        epoch_time = time.perf_counter() - start

        history["train_loss"].append(train_metrics.loss)
        history["train_acc"].append(train_metrics.accuracy)
        history["val_loss"].append(val_metrics.loss)
        history["val_acc"].append(val_metrics.accuracy)
        history["lr"].append(lr)
        history["epoch_time"].append(epoch_time)

        improved = val_metrics.loss < history["best_val_loss"]
        if improved:
            history["best_epoch"] = epoch
            history["best_val_loss"] = val_metrics.loss
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                val_metrics.loss,
                config,
            )

        if verbose:
            marker = " *" if improved else ""
            print(
                f"Época {epoch:>3}/{config.epochs} | "
                f"train loss {train_metrics.loss:.4f} "
                f"acc {train_metrics.accuracy:.4f} | "
                f"val loss {val_metrics.loss:.4f} "
                f"acc {val_metrics.accuracy:.4f} | "
                f"{epoch_time:.1f}s{marker}"
            )

    if history["best_epoch"] == 0:
        raise RuntimeError("val_loss no fue finita en ninguna época; no se guardó checkpoint.")

    return history
