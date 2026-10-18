# AJUSTES POR ETAPA (nubes CCSN):
# Editable: extensiones, política de duplicados y rutas de entrada/salida.
# Nunca sobrescribir las imágenes originales de data/raw.

"""Cura un dataset de imágenes de nubes organizado por clases.

El módulo crea una copia curada de un dataset de imágenes en bruto,
preservando los archivos originales. Detecta duplicados exactos mediante
hashes SHA-256 y aplica una política explícita:

1. Si las imágenes duplicadas pertenecen a una sola clase, se conserva una
   copia determinista.
2. Si hay imágenes idénticas en clases distintas, se excluye todo el grupo
   hasta que se pueda establecer su etiqueta correcta con evidencia fiable.
3. Si una imagen es ilegible, se excluye y se registra el motivo.
4. Se conservan todos los directorios de clase originales, aunque una clase
   quede vacía.

Example:
    uv run python -m clouds_cnn.curate_dataset \
        --input-dir data/raw \
        --output-dir data/curated \
        --report-dir reports/curation

El dataset de entrada nunca se modifica.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)

# Solo los archivos con estas extensiones se consideran imágenes candidatas.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

MANIFEST_FILENAME = "curation_manifest.csv"
EXCLUSIONS_FILENAME = "curation_exclusions.csv"
SUMMARY_FILENAME = "curation_summary.json"


def _is_image_file(path: Path) -> bool:
    """Verifica si un archivo tiene una extensión de imagen soportada.

    Args:
        path: Ruta del archivo que se va a inspeccionar.

    Returns:
        True si la ruta es un archivo con una extensión de imagen soportada.
    """
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo.

    El archivo se lee por bloques para evitar cargarlo completo en memoria.

    Args:
        path: Ruta del archivo al que se le calculará el hash.

    Returns:
        Digest SHA-256 en formato hexadecimal.
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _is_readable_image(path: Path) -> tuple[bool, str]:
    """Verifica si Pillow puede validar y decodificar una imagen.

    Args:
        path: Ruta de la imagen candidata.

    Returns:
        Una tupla con un indicador de legibilidad y un mensaje de error.
        El mensaje de error es una cadena vacía para imágenes legibles.
    """
    try:
        # verify() comprueba la integridad del archivo sin decodificar por
        # completo la imagen.
        with Image.open(path) as image:
            image.verify()

        # Se reabre la imagen porque verify() invalida el acceso normal a
        # ella.
        with Image.open(path) as image:
            image.load()

        return True, ""

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
    ) as error:
        return False, f"{type(error).__name__}: {error}"


def _validate_directories(
    input_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, list[Path]]:
    """Valida los directorios de origen y destino.

    El origen y el destino deben ser directorios separados y no anidados.
    Esto evita que la operación de salida sobrescriba o copie de forma
    recursiva el dataset de origen.

    Args:
        input_dir: Directorio raíz del dataset original.
        output_dir: Directorio raíz del dataset curado.

    Returns:
        Ruta de origen resuelta, ruta de destino resuelta y directorios de
        clase ordenados encontrados en el origen.

    Raises:
        FileNotFoundError: Si el directorio de origen no existe.
        NotADirectoryError: Si la ruta de origen no es un directorio.
        ValueError: Si el origen y el destino se superponen o no hay
            directorios de clase.
    """
    source = Path(input_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"Source dataset does not exist: {source}"
        )

    if not source.is_dir():
        raise NotADirectoryError(
            f"Source dataset is not a directory: {source}"
        )

    # Se rechazan rutas iguales o anidadas en cualquier dirección. El origen
    # y el destino deben ser hermanos, p. ej. data/raw y data/curated.
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError(
            "Source and destination must be separate, non-nested "
            f"directories. Source: {source}; destination: {destination}"
        )

    class_dirs = sorted(
        path for path in source.iterdir() if path.is_dir()
    )

    if not class_dirs:
        raise ValueError(
            f"No class directories found in source dataset: {source}"
        )

    return source, destination, class_dirs


def _scan_dataset(
    source: Path,
) -> list[dict[str, Any]]:
    """Inventaría los archivos de imagen y calcula hashes y legibilidad.

    El primer directorio de cada ruta relativa determina la clase de la
    imagen. Se admiten subdirectorios anidados dentro de una clase.

    Args:
        source: Directorio raíz resuelto del dataset original.

    Returns:
        Un registro por imagen soportada, que incluye su hash, clase, ruta
        relativa y estado de legibilidad.
    """
    image_paths = sorted(
        path for path in source.rglob("*") if _is_image_file(path)
    )

    records: list[dict[str, Any]] = []

    for path in image_paths:
        relative_path = path.relative_to(source)
        class_name = relative_path.parts[0]

        # Los errores de hash se registran en lugar de ignorarse en silencio.
        try:
            file_hash = _sha256_file(path)
            hash_error = ""
        except OSError as error:
            file_hash = ""
            hash_error = f"Hash error: {error}"

        readable, image_error = _is_readable_image(path)

        errors = [
            message
            for message in (hash_error, image_error)
            if message
        ]

        records.append(
            {
                "relative_path": relative_path.as_posix(),
                "class_name": class_name,
                "sha256": file_hash,
                "readable": readable and bool(file_hash),
                "error": " | ".join(errors),
                "action": "",
                "reason": "",
                "destination_path": "",
                "duplicate_group_size": 1,
                "duplicate_classes": class_name,
            }
        )

    LOGGER.info("Found %d supported image files.", len(records))
    return records


def _group_by_hash(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Agrupa las imágenes con hash calculado por contenido exacto de archivo.

    Args:
        records: Inventario devuelto por ``_scan_dataset``.

    Returns:
        Diccionario que asigna cada hash SHA-256 a todos los registros que
        tienen ese hash.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if record["sha256"]:
            groups[record["sha256"]].append(record)

    return dict(groups)



def _set_group_metadata(
    group: list[dict[str, Any]],
    classes: list[str],
) -> None:
    """Registra el tamaño y las clases asociadas a un grupo de duplicados.

    Args:
        group: Registros de imágenes con el mismo hash.
        classes: Nombres de las clases presentes en el grupo.
    """
    for record in group:
        record["duplicate_group_size"] = len(group)
        record["duplicate_classes"] = ";".join(classes)


def _exclude_unreadable_records(
    group: list[dict[str, Any]],
) -> None:
    """Excluye los registros de imágenes ilegibles.

    Args:
        group: Registros de imágenes con el mismo hash.
    """
    for record in group:
        if not record["readable"]:
            record["action"] = "exclude"
            record["reason"] = "unreadable_or_hash_error"


def _exclude_cross_class_duplicates(
    readable_group: list[dict[str, Any]],
) -> None:
    """Excluye duplicados legibles cuya etiqueta de clase es ambigua.

    Args:
        readable_group: Registros legibles del grupo de duplicados.
    """
    for record in readable_group:
        record["action"] = "exclude"
        record["reason"] = "cross_class_duplicate_label_unresolved"


def _decide_actions(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Asigna acciones y motivos de curación a cada registro.

    Las imágenes ilegibles se excluyen. Los duplicados entre clases
    también se excluyen, mientras que los duplicados de una misma clase
    se reducen a una copia, elegida por orden lexicográfico de su ruta.

    Args:
        records: Inventario devuelto por ``_scan_dataset``.

    Returns:
        Registros con acciones, motivos y metadatos de duplicados.
    """
    groups = _group_by_hash(records)

    for group in groups.values():
        classes = sorted({record["class_name"] for record in group})
        _set_group_metadata(group, classes)
        _exclude_unreadable_records(group)

        readable_group = [
            record for record in group if record["readable"]
        ]

        if len(classes) > 1:
            _exclude_cross_class_duplicates(readable_group)
            continue

        readable_group.sort(key=lambda record: record["relative_path"])

        if readable_group:
            keeper = readable_group[0]
            keeper["action"] = "keep"
            keeper["reason"] = (
                "unique_image"
                if len(group) == 1
                else "retained_representative_of_same_class_duplicates"
            )

            for duplicate in readable_group[1:]:
                duplicate["action"] = "exclude"
                duplicate["reason"] = "redundant_same_class_duplicate"

    for record in records:
        if not record["action"]:
            record["action"] = "exclude"
            record["reason"] = "hash_failed_or_unclassified"

    return records


def _prepare_destination(
    destination: Path,
    overwrite: bool,
) -> None:
    """Crea un directorio de salida vacío de forma segura.

    Args:
        destination: Directorio de destino de las imágenes curadas.
        overwrite: Indica si se puede reemplazar un destino existente.

    Raises:
        FileExistsError: Si el destino contiene archivos y overwrite es
            False.
    """
    if destination.exists():
        if any(destination.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Destination is not empty: {destination}. "
                "Choose another path or pass --overwrite."
            )

        if overwrite:
            # Esta función se llama solo después de validar que las rutas de
            # origen y destino son independientes y no están anidadas.
            shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)


def _copy_kept_images(
    source: Path,
    destination: Path,
    records: list[dict[str, Any]],
    class_dirs: list[Path],
) -> None:
    """Copia las imágenes conservadas y recrea todos los directorios de clase.

    Se preservan las rutas relativas originales. Las clases vacías se
    mantienen presentes para que los nombres de clase no desaparezcan en
    silencio del dataset curado.

    Args:
        source: Raíz del dataset original.
        destination: Raíz del dataset curado.
        records: Registros con las acciones asignadas por
            ``_decide_actions``.
        class_dirs: Directorios de clase originales.
    """
    # Se crean todos los directorios de clase, incluidos los que podrían
    # quedar vacíos.
    for class_dir in class_dirs:
        relative_class = class_dir.relative_to(source)
        (destination / relative_class).mkdir(parents=True, exist_ok=True)

    for record in records:
        if record["action"] != "keep":
            continue

        source_path = source / Path(record["relative_path"])
        destination_path = destination / Path(record["relative_path"])

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

        record["destination_path"] = destination_path.relative_to(
            destination
        ).as_posix()


def _write_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Escribe los registros de auditoría en un archivo CSV.

    Args:
        path: Ruta del CSV de salida.
        records: Registros que se van a serializar.
    """
    fieldnames = [
        "relative_path",
        "class_name",
        "sha256",
        "readable",
        "error",
        "action",
        "reason",
        "destination_path",
        "duplicate_group_size",
        "duplicate_classes",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _build_summary(
    source: Path,
    destination: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calcula los conteos resumen del reporte de curación.

    Args:
        source: Raíz del dataset original.
        destination: Raíz del dataset curado.
        records: Inventario con las acciones de curación.

    Returns:
        Diccionario con los totales y los conteos por motivo de exclusión.
    """
    kept = [record for record in records if record["action"] == "keep"]
    excluded = [
        record for record in records if record["action"] == "exclude"
    ]

    reason_counts: dict[str, int] = defaultdict(int)
    for record in excluded:
        reason_counts[record["reason"]] += 1

    # Se cuentan los hashes distintos solo de los grupos entre clases que
    # fueron excluidos.
    unresolved_hashes = {
        record["sha256"]
        for record in excluded
        if record["reason"] == "cross_class_duplicate_label_unresolved"
        and record["sha256"]
    }

    class_counts: dict[str, int] = defaultdict(int)
    for record in kept:
        class_counts[record["class_name"]] += 1

    return {
        "source_directory": str(source),
        "curated_directory": str(destination),
        "total_images_found": len(records),
        "images_kept": len(kept),
        "images_excluded": len(excluded),
        "excluded_by_reason": dict(sorted(reason_counts.items())),
        "unresolved_cross_class_duplicate_groups": len(unresolved_hashes),
        "curated_images_by_class": dict(sorted(class_counts.items())),
        "policy": {
            "same_class_duplicates": "Keep one deterministic copy.",
            "cross_class_duplicates": (
                "Exclude every copy until the correct label is verified."
            ),
            "unreadable_images": "Exclude and record the error.",
            "raw_dataset": "Never modified.",
        },
    }


def curate_dataset(
    input_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/curated",
    report_dir: str | Path = "reports/curation",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Crea un dataset curado y guarda un registro de auditoría.

    Args:
        input_dir: Raíz del dataset original organizado por clases.
        output_dir: Destino de los directorios de clase curados.
        report_dir: Destino del manifiesto, las exclusiones y el resumen
            JSON.
        overwrite: Si es True, reemplaza un directorio de salida existente.
            Usar con cuidado; el directorio de salida existente se eliminará.

    Returns:
        Resumen del proceso de curación.

    Raises:
        FileNotFoundError: Si el directorio de origen no existe.
        NotADirectoryError: Si la ruta de origen no es un directorio.
        ValueError: Si la estructura de origen o la relación entre
            directorios no es válida.
        FileExistsError: Si el directorio de salida no está vacío y
            overwrite es False.
    """
    source, destination, class_dirs = _validate_directories(
        input_dir,
        output_dir,
    )

    report_path = Path(report_dir).expanduser().resolve()

    # Se evita poner los reportes dentro de la jerarquía de clases curada,
    # donde un cargador de imágenes podría confundir los directorios de
    # reportes con clases.
    if report_path == destination or report_path.is_relative_to(destination):
        raise ValueError(
            "The report directory must not be inside the curated dataset."
        )

    records = _scan_dataset(source)
    records = _decide_actions(records)

    _prepare_destination(destination, overwrite)
    _copy_kept_images(source, destination, records, class_dirs)

    report_path.mkdir(parents=True, exist_ok=True)

    _write_csv(report_path / MANIFEST_FILENAME, records)
    _write_csv(
        report_path / EXCLUSIONS_FILENAME,
        [
            record
            for record in records
            if record["action"] == "exclude"
        ],
    )

    summary = _build_summary(source, destination, records)

    with (report_path / SUMMARY_FILENAME).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    LOGGER.info("Curation complete.")
    LOGGER.info("Images found: %d", summary["total_images_found"])
    LOGGER.info("Images retained: %d", summary["images_kept"])
    LOGGER.info("Images excluded: %d", summary["images_excluded"])
    LOGGER.info("Curated dataset: %s", destination)
    LOGGER.info("Curation reports: %s", report_path)

    return summary


def main() -> None:
    """Analiza los argumentos de línea de comandos y ejecuta la curación."""
    parser = argparse.ArgumentParser(
        description="Create a curated copy of a cloud-image dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Original dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/curated"),
        help="Destination for curated class directories.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/curation"),
        help="Destination for audit CSV and JSON files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory. Use with care.",
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

    curate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()