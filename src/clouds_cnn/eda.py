# AJUSTES POR ETAPA (nubes CCSN):
# Editable: diagnósticos y gráficos EDA; preservados íntegramente.

"""Análisis exploratorio de datos para un conjunto de imágenes de nubes.

Este módulo analiza imágenes organizadas en directorios por clase, como
``data/raw/Ac/``, ``data/raw/Cu/`` y ``data/raw/St/``.

El análisis es de solo lectura respecto al conjunto de datos original. Genera
archivos CSV, un resumen en JSON y, cuando Matplotlib está instalado, gráficos
de diagnóstico.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat, UnidentifiedImageError

# Se configura un logger a nivel de módulo para que el análisis pueda
# informar su progreso.
LOGGER = logging.getLogger(__name__)

# Estos son los formatos de imagen más comunes en conjuntos de datos de
# clasificación de imágenes. La lista puede ampliarse si el conjunto de datos
# contiene formatos adicionales.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# Archivos de salida generados por este módulo.
INVENTORY_FILENAME = "image_inventory.csv"
CLASS_BALANCE_FILENAME = "class_balance.csv"
IMAGE_PROPERTIES_FILENAME = "image_properties.csv"
DUPLICATES_FILENAME = "exact_duplicates.csv"
SUMMARY_FILENAME = "eda_summary.json"

# Nombres usados para los gráficos de diagnóstico opcionales.
CLASS_BALANCE_PLOT = "class_balance.png"
BRIGHTNESS_CONTRAST_PLOT = "brightness_contrast.png"
CLASS_EXAMPLES_PLOT = "class_examples.png"


def _is_image_file(path: Path) -> bool:
    """Verifica si una ruta tiene una extensión de imagen soportada.

    Args:
        path: Ruta del archivo que se está inspeccionando.

    Returns:
        True si la extensión del archivo está soportada; de lo contrario,
        False.
    """
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula el hash SHA-256 de un archivo sin cargarlo completo en memoria.

    Los hashes permiten que el análisis detecte archivos idénticos byte a
    byte, incluso cuando tienen nombres distintos o están en carpetas
    diferentes.

    Args:
        path: Ruta del archivo.
        chunk_size: Número de bytes leídos por iteración.

    Returns:
        Digest SHA-256 del archivo en formato hexadecimal.
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _get_image_metrics(path: Path) -> dict[str, Any]:
    """Extrae dimensiones, modo de color, brillo y contraste de una imagen.

    El brillo se aproxima mediante la intensidad media en escala de grises.
    El contraste se aproxima mediante la desviación estándar de la intensidad
    en escala de grises. Estas mediciones son estadísticas descriptivas, no
    medidas de calidad de imagen.

    Args:
        path: Ruta del archivo de imagen.

    Returns:
        Diccionario con las propiedades de la imagen. Si la imagen no puede
        leerse, el diccionario incluye una descripción del error y mediciones
        nulas.
    """
    result: dict[str, Any] = {
        "width": None,
        "height": None,
        "mode": None,
        "brightness": None,
        "contrast": None,
        "readable": False,
        "error": "",
    }

    try:
        # Se abre la imagen dentro de un gestor de contexto para asegurar que
        # el archivo se cierre.
        with Image.open(path) as image:
            image.verify()

        # Se vuelve a abrir el archivo porque verify() invalida las
        # operaciones normales sobre la imagen.
        with Image.open(path) as image:
            result["width"], result["height"] = image.size
            result["mode"] = image.mode

            # Se convierte una copia a escala de grises para que todas las
            # imágenes puedan compararse con la misma escala de intensidad,
            # sin importar su modo de color.
            grayscale = image.convert("L")
            image_stats = ImageStat.Stat(grayscale)

            result["brightness"] = float(image_stats.mean[0])
            result["contrast"] = float(image_stats.stddev[0])
            result["readable"] = True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
    ) as error:
        result["error"] = f"{type(error).__name__}: {error}"

    return result


def inspect_images(data_dir: str | Path) -> list[dict[str, Any]]:
    """Inspecciona todos los archivos de imagen soportados en un dataset.

    La estructura esperada es un directorio por clase. Cada imagen se asigna
    a la clase representada por su primer directorio relativo a ``data_dir``.
    Se admiten directorios anidados dentro de una clase.

    Args:
        data_dir: Directorio raíz que contiene los subdirectorios de clases.

    Returns:
        Un diccionario por imagen soportada, que incluye su ruta relativa,
        nombre de clase, propiedades de la imagen, legibilidad y hash SHA-256.

    Raises:
        FileNotFoundError: Si el directorio del dataset no existe.
        NotADirectoryError: Si ``data_dir`` no es un directorio.
        ValueError: Si no hay subdirectorios de clases.
    """
    root = Path(data_dir).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())

    if not class_dirs:
        raise ValueError(
            f"No class subdirectories were found in the dataset: {root}"
        )

    image_paths = sorted(
        path
        for path in root.rglob("*")
        if _is_image_file(path)
    )

    records: list[dict[str, Any]] = []

    for path in image_paths:
        relative_path = path.relative_to(root)
        class_name = relative_path.parts[0]
        metrics = _get_image_metrics(path)

        # Un fallo al calcular el hash no debe detener la inspección del
        # resto de los archivos.
        try:
            file_hash = _sha256_file(path)
        except OSError as error:
            file_hash = ""
            metrics["readable"] = False
            metrics["error"] = (
                f"{metrics['error']} | Hash error: {error}".strip(" |")
            )

        records.append(
            {
                "relative_path": relative_path.as_posix(),
                "class_name": class_name,
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "file_size_bytes": path.stat().st_size,
                "sha256": file_hash,
                **metrics,
            }
        )

    LOGGER.info(
        "Inspected %d image files in %d class directories.",
        len(records),
        len(class_dirs),
    )

    return records


def summarize_class_balance(
    records: list[dict[str, Any]],
    data_dir: str | Path,
) -> list[dict[str, Any]]:
    """Calcula el número y porcentaje de imágenes de cada clase.

    Los directorios de clase vacíos se incluyen en los resultados. Esto es
    importante porque una clase sin imágenes no debe desaparecer en silencio
    del reporte de balance.

    Args:
        records: Registros de imágenes devueltos por ``inspect_images``.
        data_dir: Directorio raíz que contiene los subdirectorios de clases.

    Returns:
        Una lista con el nombre de la clase, el conteo de imágenes, el
        porcentaje y el conteo de imágenes legibles de cada clase.
    """
    root = Path(data_dir).expanduser().resolve()
    class_names = sorted(
        path.name for path in root.iterdir() if path.is_dir()
    )

    counts = {class_name: 0 for class_name in class_names}
    readable_counts = {class_name: 0 for class_name in class_names}

    for record in records:
        class_name = record["class_name"]
        counts[class_name] = counts.get(class_name, 0) + 1

        if record["readable"]:
            readable_counts[class_name] = (
                readable_counts.get(class_name, 0) + 1
            )

    total_images = sum(counts.values())
    summary: list[dict[str, Any]] = []

    for class_name in class_names:
        count = counts[class_name]
        percentage = (
            100.0 * count / total_images if total_images else 0.0
        )

        summary.append(
            {
                "class_name": class_name,
                "image_count": count,
                "percentage": round(percentage, 4),
                "readable_image_count": readable_counts[class_name],
            }
        )

    return summary


def summarize_image_properties(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resume dimensiones, modos de color y estadísticas visuales de imágenes.

    Solo las imágenes legibles contribuyen a las dimensiones, modos, brillo
    y contraste. Las imágenes ilegibles se cuentan por separado.

    Args:
        records: Registros de imágenes devueltos por ``inspect_images``.

    Returns:
        Un diccionario con conteos, frecuencias de dimensiones, frecuencias
        de modos de color y estadísticas descriptivas de brillo y contraste.
    """
    readable = [record for record in records if record["readable"]]

    dimension_counts: dict[str, int] = defaultdict(int)
    mode_counts: dict[str, int] = defaultdict(int)

    for record in readable:
        dimension = f"{record['width']}x{record['height']}"
        dimension_counts[dimension] += 1
        mode_counts[record["mode"]] += 1

    brightness_values = [
        record["brightness"]
        for record in readable
        if record["brightness"] is not None
    ]
    contrast_values = [
        record["contrast"]
        for record in readable
        if record["contrast"] is not None
    ]

    def describe(values: list[float]) -> dict[str, float | None]:
        """Calcula estadísticas descriptivas básicas de una lista numérica."""
        if not values:
            return {
                "mean": None,
                "median": None,
                "minimum": None,
                "maximum": None,
                "standard_deviation": None,
            }

        return {
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
            "minimum": round(min(values), 4),
            "maximum": round(max(values), 4),
            "standard_deviation": round(
                statistics.pstdev(values), 4
            ),
        }

    return {
        "total_images": len(records),
        "readable_images": len(readable),
        "unreadable_images": len(records) - len(readable),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "color_mode_counts": dict(sorted(mode_counts.items())),
        "brightness": describe(brightness_values),
        "contrast": describe(contrast_values),
    }


def find_exact_duplicates(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Encuentra archivos idénticos byte a byte usando hashes SHA-256.

    Los duplicados exactos pueden causar fuga de datos (data leakage) si
    copias de la misma imagen quedan en distintas particiones del dataset.
    Esta función detecta contenidos de archivo idénticos, pero no detecta
    imágenes visualmente similares que hayan sido redimensionadas,
    recomprimidas o modificadas de otra forma.

    Args:
        records: Registros de imágenes devueltos por ``inspect_images``.

    Returns:
        Un registro por cada imagen que pertenece a un grupo de duplicados.
        Cada registro incluye su hash, ruta y nombre de clase.
    """
    grouped_paths: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        file_hash = record["sha256"]

        if file_hash:
            grouped_paths[file_hash].append(record)

    duplicates: list[dict[str, Any]] = []

    for file_hash, group in grouped_paths.items():
        if len(group) < 2:
            continue

        for record in group:
            duplicates.append(
                {
                    "sha256": file_hash,
                    "relative_path": record["relative_path"],
                    "class_name": record["class_name"],
                    "duplicate_group_size": len(group),
                }
            )

    return sorted(
        duplicates,
        key=lambda record: (
            record["sha256"],
            record["relative_path"],
        ),
    )


def _calculate_balance_diagnostics(
    class_balance: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calcula indicadores simples de desbalance de clases.

    La razón mayoría/minoría es el conteo de la clase más grande dividido
    entre el conteo no nulo más pequeño. Es un indicador de diagnóstico, no
    un umbral universal para decidir si un dataset está balanceado.

    Args:
        class_balance: Resultados devueltos por ``summarize_class_balance``.

    Returns:
        Diccionario con los conteos de clases, la razón y una advertencia
        sobre clases vacías.
    """
    counts = [row["image_count"] for row in class_balance]
    nonzero_counts = [count for count in counts if count > 0]

    if not counts:
        return {
            "number_of_classes": 0,
            "majority_minority_ratio": None,
            "empty_classes": [],
            "balance_note": "No classes were found.",
        }

    empty_classes = [
        row["class_name"]
        for row in class_balance
        if row["image_count"] == 0
    ]

    ratio = (
        max(nonzero_counts) / min(nonzero_counts)
        if nonzero_counts
        else None
    )

    return {
        "number_of_classes": len(class_balance),
        "majority_minority_ratio": (
            round(ratio, 4) if ratio is not None else None
        ),
        "empty_classes": empty_classes,
        "balance_note": (
            "The ratio is descriptive; assess it alongside class counts, "
            "the task and the evaluation metric."
        ),
    }


def _write_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    """Escribe una lista de diccionarios en un archivo CSV con codificación UTF-8.

    Args:
        output_path: Ruta de destino.
        rows: Filas que se van a escribir.
        fieldnames: Orden opcional de las columnas. Si se omite, la primera
            fila define las columnas.
    """
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        if not fieldnames:
            file.write("")
            return

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_class_balance_plot(
    class_balance: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Guarda un gráfico de barras con el número de imágenes por clase.

    Args:
        class_balance: Resultados devueltos por ``summarize_class_balance``.
        output_path: Ruta de destino de la imagen PNG.
    """
    import matplotlib.pyplot as plt

    names = [row["class_name"] for row in class_balance]
    counts = [row["image_count"] for row in class_balance]

    figure, axis = plt.subplots(figsize=(10, 5))
    # Un color distinto por clase
    colors = plt.cm.tab10(range(len(names)))
    axis.bar(names, counts, color=colors)
    axis.set_title("Number of images per class")
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of images")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_brightness_contrast_plot(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Guarda un diagrama de dispersión de brillo contra contraste.

    Args:
        records: Registros de imágenes devueltos por ``inspect_images``.
        output_path: Ruta de destino de la imagen PNG.
    """
    import matplotlib.pyplot as plt

    readable = [record for record in records if record["readable"]]

    brightness = [record["brightness"] for record in readable]
    contrast = [record["contrast"] for record in readable]
    labels = [record["class_name"] for record in readable]

    figure, axis = plt.subplots(figsize=(8, 6))

    # Se grafica cada clase por separado para poder observar patrones
    # específicos de cada una.
    for class_name in sorted(set(labels)):
        indices = [
            index
            for index, label in enumerate(labels)
            if label == class_name
        ]
        axis.scatter(
            [brightness[index] for index in indices],
            [contrast[index] for index in indices],
            label=class_name,
            alpha=0.7,
        )

    axis.set_title("Brightness versus contrast")
    axis.set_xlabel("Mean grayscale intensity")
    axis.set_ylabel("Grayscale intensity standard deviation")

    if labels:
        axis.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_class_examples_plot(
    data_dir: str | Path,
    class_balance: list[dict[str, Any]],
    output_path: Path,
    max_classes: int = 11,
) -> None:
    """Guarda una cuadrícula con una imagen de ejemplo de cada clase.

    Args:
        data_dir: Directorio raíz que contiene los subdirectorios de clases.
        class_balance: Resultados devueltos por ``summarize_class_balance``.
        output_path: Ruta de destino de la imagen PNG.
        max_classes: Número máximo de ejemplos de clase que se mostrarán.
    """
    import matplotlib.pyplot as plt

    root = Path(data_dir).expanduser().resolve()
    class_names = [
        row["class_name"]
        for row in class_balance
        if row["image_count"] > 0
    ][:max_classes]

    if not class_names:
        LOGGER.warning("No class examples are available to plot.")
        return

    figure, axes = plt.subplots(
        1,
        len(class_names),
        figsize=(3 * len(class_names), 3),
        squeeze=False,
    )

    for axis, class_name in zip(axes[0], class_names, strict=True):
        class_images = sorted(
            path
            for path in (root / class_name).rglob("*")
            if _is_image_file(path)
        )

        displayed = False

        for path in class_images:
            try:
                with Image.open(path) as image:
                    image.load()
                    axis.imshow(image.convert("RGB"))
                axis.set_title(class_name)
                displayed = True
                break
            except (OSError, ValueError, UnidentifiedImageError):
                # Se continúa hasta encontrar un ejemplo legible.
                continue

        if not displayed:
            axis.set_title(f"{class_name}\n(no readable image)")

        axis.axis("off")

    figure.suptitle("Example image from each class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_eda_report(
    data_dir: str | Path,
    output_dir: str | Path,
    records: list[dict[str, Any]],
    class_balance: list[dict[str, Any]],
    properties: dict[str, Any],
    duplicates: list[dict[str, Any]],
    make_plots: bool = True,
) -> dict[str, Any]:
    """Guarda reportes CSV, un resumen JSON y gráficos de diagnóstico opcionales.

    Args:
        data_dir: Directorio raíz que contiene las imágenes originales.
        output_dir: Directorio donde se guardarán los artefactos del reporte.
        records: Inventario de imágenes devuelto por ``inspect_images``.
        class_balance: Conteos de clases devueltos por
            ``summarize_class_balance``.
        properties: Resumen devuelto por ``summarize_image_properties``.
        duplicates: Resultados devueltos por ``find_exact_duplicates``.
        make_plots: Indica si se intentará generar gráficos con Matplotlib.

    Returns:
        El mismo diccionario de resumen escrito en ``eda_summary.json``.
    """
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    _write_csv(destination / INVENTORY_FILENAME, records)
    _write_csv(destination / CLASS_BALANCE_FILENAME, class_balance)
    _write_csv(
        destination / IMAGE_PROPERTIES_FILENAME,
        [
            record
            for record in records
            if record["readable"]
        ],
    )
    _write_csv(destination / DUPLICATES_FILENAME, duplicates)

    # Se incluyen indicadores de calidad de datos junto con las estadísticas
    # descriptivas.
    summary = {
        "dataset_directory": str(Path(data_dir).expanduser().resolve()),
        "report_directory": str(destination),
        "properties": properties,
        "class_balance": class_balance,
        "balance_diagnostics": _calculate_balance_diagnostics(
            class_balance
        ),
        "exact_duplicate_groups": len(
            {
                record["sha256"]
                for record in duplicates
            }
        ),
        "images_in_exact_duplicate_groups": len(duplicates),
        "unreadable_images": [
            record["relative_path"]
            for record in records
            if not record["readable"]
        ],
    }

    summary_path = destination / SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    if make_plots:
        try:
            _save_class_balance_plot(
                class_balance,
                destination / CLASS_BALANCE_PLOT,
            )
            _save_brightness_contrast_plot(
                records,
                destination / BRIGHTNESS_CONTRAST_PLOT,
            )
            _save_class_examples_plot(
                data_dir,
                class_balance,
                destination / CLASS_EXAMPLES_PLOT,
            )
        except ImportError:
            LOGGER.warning(
                "Matplotlib is not installed. CSV and JSON reports were "
                "created, but plots were skipped. Install it with "
                "'uv add matplotlib' if plots are needed."
            )

    LOGGER.info("EDA reports saved to %s", destination)
    return summary


def run_eda(
    data_dir: str | Path,
    output_dir: str | Path,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Ejecuta el flujo completo de análisis exploratorio.

    Args:
        data_dir: Directorio que contiene un subdirectorio por cada clase de
            imagen.
        output_dir: Directorio de destino para los reportes CSV, JSON y PNG.
        make_plots: Indica si se generan gráficos cuando Matplotlib está
            disponible.

    Returns:
        Diccionario de resumen con el balance de clases, las propiedades de
        las imágenes, la información de duplicados y la información de
        archivos ilegibles.

    Raises:
        FileNotFoundError: Si el directorio del dataset no existe.
        NotADirectoryError: Si la ruta del dataset no es un directorio.
        ValueError: Si el dataset no tiene subdirectorios de clases.
    """
    records = inspect_images(data_dir)
    class_balance = summarize_class_balance(records, data_dir)
    properties = summarize_image_properties(records)
    duplicates = find_exact_duplicates(records)

    summary = save_eda_report(
        data_dir=data_dir,
        output_dir=output_dir,
        records=records,
        class_balance=class_balance,
        properties=properties,
        duplicates=duplicates,
        make_plots=make_plots,
    )

    # Se registran los hallazgos principales para que el script sea útil
    # desde la terminal.
    diagnostics = summary["balance_diagnostics"]
    LOGGER.info("Total images: %s", properties["total_images"])
    LOGGER.info("Readable images: %s", properties["readable_images"])
    LOGGER.info("Unreadable images: %s", properties["unreadable_images"])
    LOGGER.info("Number of classes: %s", diagnostics["number_of_classes"])
    LOGGER.info(
        "Majority/minority ratio: %s",
        diagnostics["majority_minority_ratio"],
    )
    LOGGER.info(
        "Images belonging to exact-duplicate groups: %s",
        summary["images_in_exact_duplicate_groups"],
    )

    return summary


def main() -> None:
    """Analiza los argumentos de línea de comandos y ejecuta el análisis EDA."""
    parser = argparse.ArgumentParser(
        description="Perform exploratory analysis on a class-organized "
        "image dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Dataset root containing one directory per class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/eda"),
        help="Directory where reports and plots will be saved.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Generate CSV and JSON reports without attempting plots.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display informational logging messages.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    run_eda(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()