# AJUSTES POR ETAPA (nubes CCSN):
# Editable: métricas y validaciones. La matriz se parametriza por num_classes=11;
# los ejemplos unitarios pequeños de 3 clases siguen siendo válidos.

"""Métricas de clasificación y generalization gap.

Todo se calcula con tensores de PyTorch a partir de etiquetas enteras (N,).
Este módulo no conoce MLflow ni Matplotlib: devuelve números y tensores.

Convención de la matriz de confusión:
cm[i, j] = muestras de clase real i predichas como clase j.

Las divisiones por cero devuelven 0, igual que zero_division=0
en scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from clouds_cnn.engine import evaluate


@dataclass(frozen=True)
class Predictions:
    """Predicciones de un conjunto de datos, en CPU y orden del loader.

    Attributes:
        y_true: Etiquetas reales (N,), tipo long.
        y_pred: Clases predichas (N,), tipo long.
        probabilities: Probabilidades softmax (N, K).
        loss: Pérdida promedio por muestra.
        accuracy: Fracción de aciertos.
    """

    y_true: torch.Tensor
    y_pred: torch.Tensor
    probabilities: torch.Tensor
    loss: float
    accuracy: float


@dataclass(frozen=True)
class ClassificationReport:
    """Métricas de clasificación de un conjunto de datos.

    Attributes:
        accuracy: Fracción de aciertos, en [0, 1].
        confusion_matrix: Matriz (K, K), filas reales y columnas predichas.
        precision: Precision por clase, (K,).
        recall: Recall por clase, (K,).
        f1: F1 por clase, (K,).
        macro_f1: Promedio no ponderado del F1 por clase.
    """

    accuracy: float
    confusion_matrix: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    macro_f1: float


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Predictions:
    """Recopila etiquetas, predicciones y probabilidades mediante evaluate.

    El loader debe utilizar shuffle=False si se necesita conservar
    la correspondencia entre los índices y las muestras del dataset.

    Args:
        model: Modelo que produce logits de forma (B, K).
        loader: Loader de imágenes y etiquetas.
        loss_fn: Pérdida con reduction='mean'.
        device: Dispositivo del modelo.

    Returns:
        Predicciones y métricas del conjunto evaluado.

    Raises:
        ValueError: Si el loader está vacío o la reducción no es mean.
    """
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []

    def store(logits: torch.Tensor, y_batch: torch.Tensor) -> None:
        logits_batches.append(logits)
        label_batches.append(y_batch)

    metrics = evaluate(
        model,
        loader,
        loss_fn,
        device,
        on_batch=store,
    )

    with torch.inference_mode():
        logits = torch.cat(logits_batches).float().cpu()
        y_true = torch.cat(label_batches).long().cpu()
        probabilities = torch.softmax(logits, dim=1)

    return Predictions(
        y_true=y_true,
        y_pred=logits.argmax(dim=1),
        probabilities=probabilities,
        loss=metrics.loss,
        accuracy=metrics.accuracy,
    )


def misclassified_indices(
    predictions: Predictions,
    max_count: int | None = None,
) -> torch.Tensor:
    """Obtiene errores ordenados por confianza descendente.

    Args:
        predictions: Predicciones recopiladas.
        max_count: Máximo de índices, o None para devolver todos.

    Returns:
        Índices long en CPU de las muestras mal clasificadas.

    Raises:
        ValueError: Si max_count es negativo.
    """
    if max_count is not None and max_count < 0:
        raise ValueError(
            f"max_count debe ser >= 0 o None; se recibió {max_count}"
        )

    wrong = torch.nonzero(
        predictions.y_pred != predictions.y_true
    ).flatten()

    confidence = predictions.probabilities[
        wrong,
        predictions.y_pred[wrong],
    ]

    ordered = wrong[
        torch.argsort(
            confidence,
            descending=True,
            stable=True,
        )
    ]

    return ordered if max_count is None else ordered[:max_count]


def _check_labels(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int | None,
) -> None:
    """Valida las etiquetas y, si se indica, su rango."""
    if num_classes is not None and num_classes < 1:
        raise ValueError(
            f"num_classes debe ser >= 1; se recibió {num_classes}"
        )

    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        if (
            y.ndim != 1
            or y.dtype.is_floating_point
            or y.dtype == torch.bool
        ):
            raise ValueError(
                f"{name} debe ser un tensor entero 1D; "
                f"tiene {y.dtype}, {y.shape}"
            )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true {tuple(y_true.shape)} y "
            f"y_pred {tuple(y_pred.shape)} difieren"
        )

    if y_true.numel() == 0:
        raise ValueError("y_true y y_pred están vacíos")

    if num_classes is None:
        return

    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        if y.min() < 0 or y.max() >= num_classes:
            raise ValueError(
                f"{name} tiene etiquetas fuera de [0, {num_classes})"
            )


def accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """Calcula la fracción de predicciones correctas."""
    _check_labels(y_true, y_pred, num_classes=None)
    return (y_true == y_pred).float().mean().item()


def confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Calcula la matriz de confusión (K, K).

    Las filas representan las clases reales y las columnas,
    las clases predichas.
    """
    _check_labels(y_true, y_pred, num_classes)

    codes = y_true.long() * num_classes + y_pred.long()
    counts = torch.bincount(
        codes,
        minlength=num_classes * num_classes,
    )

    return counts.reshape(num_classes, num_classes)


def _safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Divide elemento a elemento y devuelve 0 ante denominadores cero."""
    numerator = numerator.double()
    denominator = denominator.double()

    positive = denominator > 0
    ratio = numerator / torch.where(
        positive,
        denominator,
        1.0,
    )

    return torch.where(positive, ratio, 0.0)


def per_class_metrics(
    cm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcula precision, recall y F1 para cada clase."""
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(
            f"cm debe ser cuadrada (K, K); "
            f"tiene forma {tuple(cm.shape)}"
        )

    if (cm < 0).any():
        raise ValueError("cm tiene entradas negativas")

    true_positives = cm.diagonal()

    precision = _safe_divide(
        true_positives,
        cm.sum(dim=0),
    )
    recall = _safe_divide(
        true_positives,
        cm.sum(dim=1),
    )
    f1 = _safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return precision, recall, f1


def macro_f1(cm: torch.Tensor) -> float:
    """Calcula el promedio no ponderado del F1 por clase."""
    _, _, f1 = per_class_metrics(cm)
    return f1.mean().item()


def classification_report(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> ClassificationReport:
    """Calcula las métricas de clasificación a partir de las etiquetas."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return report_from_confusion_matrix(cm)


def report_from_confusion_matrix(
    cm: torch.Tensor,
) -> ClassificationReport:
    """Construye el reporte a partir de una matriz de confusión."""
    cm = cm.cpu()
    precision, recall, f1 = per_class_metrics(cm)

    if cm.sum() == 0:
        raise ValueError("cm no tiene muestras")

    return ClassificationReport(
        accuracy=(cm.diagonal().sum() / cm.sum()).item(),
        confusion_matrix=cm,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=f1.mean().item(),
    )


def generalization_gap(
    train_acc_eval: float,
    val_acc: float,
) -> float:
    """Calcula train accuracy en modo eval menos validation accuracy."""
    for name, value in (
        ("train_acc_eval", train_acc_eval),
        ("val_acc", val_acc),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} debe estar en [0, 1]; se recibió {value}"
            )

    return train_acc_eval - val_acc
