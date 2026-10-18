# AJUSTES POR ETAPA (nubes CCSN):
# Editable: SPLIT_FRACTIONS, split_seed, image_size=256 y center_crop.
# La entrada es data/curated, NO un archivo HDF5 ni descarga externa.


"""Preparación de un conjunto de imágenes de nubes para clouds_cnn.

Lee imágenes organizadas en carpetas por clase, como
``data/curated/<clase>/``, y genera las particiones ``train/``, ``val/``
y ``test/`` compatibles con ``torchvision.datasets.ImageFolder``.

También crea ``manifest.csv`` y ``dataset_stats.json``, utilizados por
el módulo ``data.py`` para conservar la trazabilidad y validar el conjunto.

Las imágenes de origen no se modifican ni se recomprimen.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

from clouds_cnn.data import (
    DATASET_STATS_NAME,
    build_base_transform,
    compute_channel_stats,
)

LOGGER = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT_FRACTIONS: dict[str, float] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
MANIFEST_NAME = "manifest.csv"
STATS_NAME = DATASET_STATS_NAME


def class_to_label(class_names: list[str]) -> dict[str, int]:
    """Asigna etiquetas enteras a las clases en orden alfabético.

    Args:
        class_names: Nombres de las carpetas de clase.

    Returns:
        Diccionario que relaciona cada clase con su etiqueta.
    """
    return {
        name: label
        for label, name in enumerate(sorted(class_names))
    }


def check_fractions(fractions: dict[str, float]) -> None:
    """Comprueba que las proporciones de las particiones sean válidas.

    Args:
        fractions: Proporciones de entrenamiento, validación y prueba.

    Raises:
        ValueError: Si las claves o los valores son inválidos.
    """
    if set(fractions) != set(SPLITS):
        raise ValueError(
            f"Se requieren las particiones {SPLITS}; se recibió {fractions}."
        )

    values = list(fractions.values())
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 < value < 1.0
        for value in values
    ):
        raise ValueError(
            f"Cada proporción debe ser finita y estar entre 0 y 1: {fractions}"
        )

    if not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Las proporciones deben sumar 1; se recibió {fractions}."
        )


def validate_parameters(
    image_size: int,
    center_crop: int | None,
    num_workers: int,
    split_seed: int,
) -> None:
    """Valida los parámetros numéricos de la preparación.

    Args:
        image_size: Tamaño de imagen utilizado en las transformaciones.
        center_crop: Tamaño opcional del recorte central.
        num_workers: Número de procesos auxiliares.
        split_seed: Semilla de la partición.

    Raises:
        ValueError: Si algún parámetro está fuera del rango permitido.
    """
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size debe ser un entero positivo.")

    if center_crop is not None:
        if not isinstance(center_crop, int) or center_crop <= 0:
            raise ValueError("center_crop debe ser un entero positivo o None.")
        if center_crop < image_size:
            raise ValueError(
                "center_crop debe ser mayor o igual que image_size."
            )

    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers debe ser un entero no negativo.")

    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("split_seed debe ser un entero.")


def validate_paths(source: Path, output: Path) -> None:
    """Comprueba que las rutas de entrada y salida sean independientes.

    Args:
        source: Carpeta con las imágenes originales.
        output: Carpeta donde se generarán las particiones.

    Raises:
        FileNotFoundError: Si la entrada no existe.
        ValueError: Si las rutas coinciden o una contiene a la otra.
    """
    if not source.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de entrada: {source}"
        )

    if source == output:
        raise ValueError("La entrada y la salida no pueden ser la misma carpeta.")

    if source in output.parents or output in source.parents:
        raise ValueError(
            "La entrada y la salida no pueden estar una dentro de la otra."
        )


def split_indices(
    labels: np.ndarray,
    split_seed: int,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> dict[str, np.ndarray]:
    """Divide los índices en tres particiones estratificadas y reproducibles.

    Args:
        labels: Etiqueta de clase de cada imagen.
        split_seed: Semilla para reproducir la partición.
        fractions: Proporciones de entrenamiento, validación y prueba.

    Returns:
        Diccionario con los índices de cada partición, ordenados.

    Raises:
        ValueError: Si no hay suficientes imágenes por clase para estratificar.
    """
    check_fractions(fractions)
    labels = np.asarray(labels)

    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Se necesita un vector de etiquetas no vacío.")

    _, class_counts = np.unique(labels, return_counts=True)
    if np.any(class_counts < 3):
        raise ValueError(
            "Cada clase necesita al menos 3 imágenes para aparecer "
            "en las tres particiones."
        )

    indices = np.arange(len(labels))
    rest_fraction = fractions["val"] + fractions["test"]
    test_within_rest = fractions["test"] / rest_fraction

    try:
        train, rest = train_test_split(
            indices,
            test_size=rest_fraction,
            stratify=labels,
            random_state=split_seed,
        )
        val, test = train_test_split(
            rest,
            test_size=test_within_rest,
            stratify=labels[rest],
            random_state=split_seed,
        )
    except ValueError as error:
        raise ValueError(
            "No fue posible crear las particiones estratificadas. "
            "Comprueba la cantidad de imágenes de cada clase y las proporciones."
        ) from error

    return {
        "train": np.sort(train),
        "val": np.sort(val),
        "test": np.sort(test),
    }


def build_manifest(
    dataset: ImageFolder,
    splits: dict[str, np.ndarray],
    source_root: Path,
) -> pd.DataFrame:
    """Registra la clase, el origen y la partición de cada imagen.

    Args:
        dataset: Conjunto de imágenes leído mediante ImageFolder.
        splits: Índices asignados a cada partición.
        source_root: Raíz de las imágenes de entrada.

    Returns:
        Tabla de trazabilidad con una fila por imagen.
    """
    rows: list[dict[str, Any]] = []

    for split in SPLITS:
        for index in splits[split]:
            source_path, label = dataset.samples[int(index)]
            relative_source = Path(source_path).relative_to(source_root)
            class_name = dataset.classes[int(label)]
            relative_output = (
                Path(split)
                / class_name
                / relative_source.relative_to(class_name)
            )
            rows.append(
                {
                    "source_relpath": relative_source.as_posix(),
                    "split": split,
                    "class_name": class_name,
                    "label": int(label),
                    "relpath": relative_output.as_posix(),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "source_relpath",
            "split",
            "class_name",
            "label",
            "relpath",
        ],
    )


def export_images(
    manifest: pd.DataFrame,
    source_root: Path,
    output_root: Path,
    class_names: list[str],
) -> None:
    """Copia las imágenes y crea las carpetas de todas las clases.

    Args:
        manifest: Registro de imágenes y particiones.
        source_root: Carpeta de las imágenes originales.
        output_root: Carpeta temporal de salida.
        class_names: Clases detectadas en el conjunto de entrada.
    """
    for split in SPLITS:
        for class_name in class_names:
            (output_root / split / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )

    for row in manifest.itertuples(index=False):
        source_path = source_root / row.source_relpath
        output_path = output_root / row.relpath
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)


def build_stats(
    manifest: pd.DataFrame,
    class_names: list[str],
    split_seed: int,
    image_size: int,
    center_crop: int | None,
    mean_train: list[float],
    std_train: list[float],
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> dict[str, Any]:
    """Construye los metadatos utilizados por el módulo data.py.

    Args:
        manifest: Registro de las imágenes particionadas.
        class_names: Nombres de las clases.
        split_seed: Semilla utilizada para dividir los datos.
        image_size: Tamaño de imagen usado para calcular las estadísticas.
        center_crop: Tamaño del recorte central, si se utiliza.
        mean_train: Media RGB calculada sobre entrenamiento.
        std_train: Desviación estándar RGB calculada sobre entrenamiento.
        fractions: Proporciones de las particiones.

    Returns:
        Diccionario con clases, cantidades, transformaciones y estadísticas.
    """
    mapping = class_to_label(class_names)
    grouped = manifest.groupby(["split", "class_name"]).size()

    counts = {
        split: {
            name: int(grouped.get((split, name), 0))
            for name in sorted(class_names)
        }
        for split in SPLITS
    }

    return {
        "split_seed": split_seed,
        "split_fractions": dict(fractions),
        "classes": [
            {"class_name": name, "label": mapping[name]}
            for name in sorted(class_names)
        ],
        "class_to_label": mapping,
        "counts": counts,
        "total": int(len(manifest)),
        "transform": {
            "image_size": image_size,
            "center_crop": center_crop,
            "ops": (
                (["CenterCrop"] if center_crop is not None else [])
                + ["Resize", "ToTensor"]
            ),
        },
        "mean_train": mean_train,
        "std_train": std_train,
    }


def resolve_output(output: Path, overwrite: bool) -> Path:
    """Prepara una ruta temporal y protege los resultados existentes.

    Args:
        output: Carpeta final solicitada.
        overwrite: Indica si se permite reemplazar una salida anterior.

    Returns:
        Ruta temporal terminada en ``.partial``.

    Raises:
        FileExistsError: Si la salida existe y no se autoriza reemplazarla.
    """
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} ya existe. Utiliza overwrite=True para regenerarla."
        )

    if output.exists() and not (output / MANIFEST_NAME).is_file():
        raise FileExistsError(
            f"{output} no contiene {MANIFEST_NAME}; "
            "no se eliminará por seguridad."
        )

    partial = output.with_name(output.name + ".partial")

    if partial.exists():
        if not partial.is_dir():
            raise FileExistsError(
                f"La ruta temporal existe y no es una carpeta: {partial}"
            )
        shutil.rmtree(partial)

    return partial


def finalize(partial: Path, output: Path) -> None:
    """Publica la carpeta temporal cuando la preparación termina correctamente.

    Args:
        partial: Carpeta temporal ya preparada.
        output: Carpeta de destino final.
    """
    if output.exists():
        shutil.rmtree(output)

    partial.rename(output)


def _check_output_labels(
    output: Path,
    class_names: list[str],
) -> None:
    """Comprueba que ImageFolder asigne las mismas etiquetas en cada partición.

    Args:
        output: Carpeta que contiene train, val y test.
        class_names: Clases detectadas en el conjunto original.

    Raises:
        ValueError: Si el mapeo de clases difiere del esperado.
    """
    expected = class_to_label(class_names)

    for split in SPLITS:
        found = ImageFolder(output / split).class_to_idx
        if found != expected:
            raise ValueError(
                f"El mapeo de clases de {split} es {found}; "
                f"se esperaba {expected}."
            )


def _calculate_train_stats(
    output: Path,
    image_size: int,
    center_crop: int | None,
    num_workers: int,
) -> tuple[list[float], list[float]]:
    """Calcula la media y la desviación estándar usando solo train.

    Args:
        output: Carpeta que contiene las particiones.
        image_size: Tamaño utilizado en la transformación.
        center_crop: Recorte central opcional.
        num_workers: Número de procesos auxiliares.

    Returns:
        Tupla con la media RGB y la desviación estándar RGB.
    """
    transform = build_base_transform(image_size, center_crop)
    train_dataset = ImageFolder(
        output / "train",
        transform=transform,
    )

    return compute_channel_stats(
        train_dataset,
        num_workers=num_workers,
    )


def _write_metadata(
    output: Path,
    manifest: pd.DataFrame,
    stats: dict[str, Any],
) -> None:
    """Guarda el manifiesto CSV y las estadísticas JSON.

    Args:
        output: Carpeta temporal de preparación.
        manifest: Registro de trazabilidad.
        stats: Metadatos serializables en formato JSON.
    """
    manifest.to_csv(output / MANIFEST_NAME, index=False)
    (output / STATS_NAME).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_to_directory(
    dataset: ImageFolder,
    source: Path,
    output: Path,
    partial: Path,
    splits: dict[str, np.ndarray],
    split_seed: int,
    image_size: int,
    center_crop: int | None,
    num_workers: int,
    fractions: dict[str, float],
) -> dict[str, Any]:
    """Ejecuta la preparación dentro de una carpeta temporal.

    Args:
        dataset: Conjunto original leído con ImageFolder.
        source: Raíz de las imágenes originales.
        output: Destino final.
        partial: Destino temporal.
        splits: Índices de cada partición.
        split_seed: Semilla utilizada.
        image_size: Tamaño para calcular las estadísticas.
        center_crop: Recorte central opcional.
        num_workers: Número de procesos auxiliares.
        fractions: Proporciones de las particiones.

    Returns:
        Estadísticas y metadatos de la preparación.
    """
    try:
        manifest = build_manifest(dataset, splits, source)
        export_images(manifest, source, partial, dataset.classes)
        _check_output_labels(partial, dataset.classes)

        mean_train, std_train = _calculate_train_stats(
            partial,
            image_size,
            center_crop,
            num_workers,
        )
        stats = build_stats(
            manifest,
            dataset.classes,
            split_seed,
            image_size,
            center_crop,
            mean_train,
            std_train,
            fractions,
        )

        _write_metadata(partial, manifest, stats)
        finalize(partial, output)
        LOGGER.info("Preparación completada en %s", output)
        return stats
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise


def prepare(
    source: Path,
    output: Path,
    split_seed: int = 2026,
    image_size: int = 256,
    center_crop: int | None = None,
    overwrite: bool = False,
    num_workers: int = 0,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> dict[str, Any]:
    """Prepara las imágenes de nubes y genera train, val y test.

    Args:
        source: Carpeta de entrada organizada por clases.
        output: Carpeta final para las particiones y los metadatos.
        split_seed: Semilla de la división reproducible.
        image_size: Tamaño de imagen usado para calcular estadísticas.
        center_crop: Recorte central opcional.
        overwrite: Permite reemplazar una salida previamente generada.
        num_workers: Número de procesos auxiliares para calcular estadísticas.
        fractions: Proporciones de entrenamiento, validación y prueba.

    Returns:
        Contenido de dataset_stats.json.

    Raises:
        FileNotFoundError: Si no existe la carpeta de entrada.
        ValueError: Si los parámetros o los datos no permiten la preparación.
        FileExistsError: Si la salida existente no puede reemplazarse.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()

    validate_paths(source, output)
    validate_parameters(image_size, center_crop, num_workers, split_seed)
    check_fractions(fractions)

    dataset = ImageFolder(source)
    if not dataset.classes or not dataset.samples:
        raise ValueError(
            f"No se encontraron imágenes organizadas por clase en {source}."
        )

    labels = np.asarray(dataset.targets)
    splits = split_indices(labels, split_seed, fractions)
    partial = resolve_output(output, overwrite)

    LOGGER.info(
        "Preparando %d imágenes de %d clases.",
        len(dataset.samples),
        len(dataset.classes),
    )
    return _prepare_to_directory(
        dataset=dataset,
        source=source,
        output=output,
        partial=partial,
        splits=splits,
        split_seed=split_seed,
        image_size=image_size,
        center_crop=center_crop,
        num_workers=num_workers,
        fractions=fractions,
    )


def main() -> None:
    """Permite ejecutar la preparación desde la terminal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepara el conjunto de imágenes de nubes."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/curated"),
        help="Carpeta de entrada organizada por clases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Carpeta de salida para las particiones.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2026,
        help="Semilla para la división reproducible.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Tamaño de imagen para calcular las estadísticas.",
    )
    parser.add_argument(
        "--center-crop",
        type=int,
        default=None,
        help="Tamaño opcional del recorte central.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Número de procesos auxiliares.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar una preparación anterior.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra mensajes informativos durante la preparación.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    stats = prepare(
        source=args.source,
        output=args.output,
        split_seed=args.split_seed,
        image_size=args.image_size,
        center_crop=args.center_crop,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
    )
    LOGGER.info("Total de imágenes preparadas: %d", stats["total"])


if __name__ == "__main__":
    main()
