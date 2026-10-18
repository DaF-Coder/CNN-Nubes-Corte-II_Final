# AJUSTES POR ETAPA (nubes CCSN):
# Editable: run_name, split_seed, train_seed, num_classes=11, image_size=256, batch_size, epochs, learning_rate, BatchNorm, Dropout y rutas.
# Mantener num_classes=11 e image_size=256 para los experimentos reales CCSN.

"""Configuración y validación de experimentos de clasificación de nubes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from clouds_cnn.utils import PROJECT_ROOT, VALID_DEVICE_PREFERENCES

# Atributos que representan rutas y deben serializarse como texto.
_PATH_FIELDS: tuple[str, ...] = ("data_root", "output_root")


@dataclass(frozen=True)
class ExperimentConfig:
    """Almacena los parámetros de un experimento de clasificación de nubes.

    La configuración centraliza los parámetros del conjunto de datos,
    la arquitectura, el entrenamiento y la reproducibilidad experimental.

    Attributes:
        run_name: Nombre que identifica la ejecución del experimento.
        split_seed: Semilla utilizada para dividir los datos.
        train_seed: Semilla utilizada durante el entrenamiento.
        deterministic: Indica si se prioriza la ejecución determinista.
        num_classes: Número de clases de nubes que se van a clasificar.
        image_size: Tamaño de las imágenes de entrada, en píxeles.
        center_crop: Tamaño del recorte central, o None si no se utiliza.
        batch_size: Número de imágenes procesadas por lote.
        num_workers: Número de procesos para cargar los datos.
        use_batchnorm: Indica si se utiliza normalización por lotes.
        dropout_p: Probabilidad de desactivación de neuronas mediante dropout.
        learning_rate: Tasa de aprendizaje del optimizador.
        weight_decay: Coeficiente de regularización L2 del optimizador.
        epochs: Número máximo de épocas de entrenamiento.
        device_preference: Dispositivo preferido para ejecutar el modelo.
        data_root: Directorio raíz de los datos de entrada.
        output_root: Directorio raíz de los resultados.
        mlflow_experiment_name: Nombre del experimento registrado en MLflow.
    """

    run_name: str = "baseline"
    split_seed: int = 2026
    train_seed: int = 2026
    deterministic: bool = False
    num_classes: int = 11
    image_size: int = 256 
    center_crop: int | None = None
    batch_size: int = 32
    num_workers: int = 0
    use_batchnorm: bool = False
    dropout_p: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 30

    device_preference: str = "auto"

    data_root: Path = field(
    default_factory=lambda: PROJECT_ROOT / "data" / "processed"
    )
    output_root: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs"
    )

    mlflow_experiment_name: str = "clouds-cnn-from-scratch"

    def __post_init__(self) -> None:
        for name in _PATH_FIELDS:
            object.__setattr__(self, name, Path(getattr(self, name)))
        errors = [message for invalid, message in self._checks() if invalid]
        if errors:
            raise ValueError("ExperimentConfig inválida: " + "; ".join(errors))

    def _checks(self) -> list[tuple[bool, str]]:
        return [
            (not self.run_name.strip(), "run_name no puede estar vacío"),
            (self.split_seed < 0, f"split_seed debe ser >= 0 (es {self.split_seed})"),
            (self.train_seed < 0, f"train_seed debe ser >= 0 (es {self.train_seed})"),
            (self.num_classes < 2, f"num_classes debe ser >= 2 (es {self.num_classes})"),
            (self.image_size < 1, f"image_size debe ser >= 1 (es {self.image_size})"),
            (
                self.center_crop is not None and self.center_crop < 1,
                f"center_crop debe ser None o >= 1 (es {self.center_crop})",
            ),
            (self.batch_size < 1, f"batch_size debe ser >= 1 (es {self.batch_size})"),
            (self.num_workers < 0, f"num_workers debe ser >= 0 (es {self.num_workers})"),
            (
                not 0.0 <= self.dropout_p < 1.0,
                f"dropout_p debe estar en [0, 1) (es {self.dropout_p})",
            ),
            (self.learning_rate <= 0, f"learning_rate debe ser > 0 (es {self.learning_rate})"),
            (self.weight_decay < 0, f"weight_decay debe ser >= 0 (es {self.weight_decay})"),
            (self.epochs < 1, f"epochs debe ser >= 1 (es {self.epochs})"),
            (
                self.device_preference not in VALID_DEVICE_PREFERENCES,
                f"device_preference debe ser uno de {VALID_DEVICE_PREFERENCES} "
                f"(es {self.device_preference!r})",
            ),
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serializa a un ``dict`` de tipos primitivos (``Path`` -> ``str``).

        Apto para el checkpoint (``torch.load(..., weights_only=True)``) y para MLflow.
        """
        data = asdict(self)
        for name in _PATH_FIELDS:
            data[name] = str(data[name])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Reconstruye la configuración desde ``to_dict()`` (por ejemplo, desde un checkpoint).

        Raises:
            ValueError: Si ``data`` contiene claves desconocidas o valores inválidos.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Claves desconocidas en la configuración: {sorted(unknown)}")
        return cls(**data)
