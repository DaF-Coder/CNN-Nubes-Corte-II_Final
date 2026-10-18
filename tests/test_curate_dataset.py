"""Pruebas unitarias del módulo de curación de datasets de imágenes de nubes.

Las pruebas crean datasets temporales para validar la política de curación
sin modificar ni depender del dataset real.

Ejecutar desde la raíz del proyecto:
    uv run pytest tests/test_curate_dataset.py -v
"""

import csv
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from clouds_cnn.curate_dataset import (
    EXCLUSIONS_FILENAME,
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    curate_dataset,
)


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Crea un dataset temporal con tres clases de imágenes.

    Args:
        tmp_path: Directorio temporal proporcionado por pytest.

    Returns:
        Raíz del dataset con los directorios de clase Ac, As y Cu.
    """
    dataset = tmp_path / "data" / "raw"

    for class_name in ("Ac", "As", "Cu"):
        (dataset / class_name).mkdir(parents=True)

    # Se crean contenidos de imagen distintos para que los archivos sean
    # únicos al inicio.
    for index, class_name in enumerate(("Ac", "As", "Cu")):
        image = Image.new(
            "RGB",
            (16, 16),
            color=(40 + index * 60, 30 + index * 20, 100),
        )
        image.save(dataset / class_name / f"{class_name}-unique.png")

    return dataset


def _copy_exact_duplicate(
    source: Path,
    destination: Path,
) -> None:
    """Crea un duplicado idéntico byte a byte de una imagen.

    Args:
        source: Imagen existente.
        destination: Ruta de destino del duplicado.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Lee un archivo CSV UTF-8 como una lista de diccionarios.

    Args:
        path: Archivo CSV que se va a leer.

    Returns:
        Lista de filas con los nombres de columna como claves del
        diccionario.
    """
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_keeps_unique_images(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que todas las imágenes únicas y legibles se copien."""
    output = tmp_path / "data" / "curated"

    summary = curate_dataset(
        input_dir=sample_dataset,
        output_dir=output,
        report_dir=tmp_path / "reports",
    )

    assert summary["total_images_found"] == 3
    assert summary["images_kept"] == 3
    assert summary["images_excluded"] == 0

    for class_name in ("Ac", "As", "Cu"):
        assert (
            output / class_name / f"{class_name}-unique.png"
        ).is_file()


def test_same_class_duplicates_keep_only_one_copy(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que los duplicados dentro de una clase se deduplican."""
    original = sample_dataset / "Ac" / "Ac-unique.png"

    _copy_exact_duplicate(
        original,
        sample_dataset / "Ac" / "Ac-copy.png",
    )

    output = tmp_path / "data" / "curated"

    summary = curate_dataset(
        input_dir=sample_dataset,
        output_dir=output,
        report_dir=tmp_path / "reports",
    )

    assert summary["total_images_found"] == 4
    assert summary["images_kept"] == 3
    assert summary["images_excluded"] == 1

    # La ruta que es primera en orden lexicográfico es el representante
    # determinista.
    assert (output / "Ac" / "Ac-copy.png").is_file()
    assert not (output / "Ac" / "Ac-unique.png").is_file()


def test_cross_class_duplicates_are_all_excluded(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que las etiquetas en conflicto no se resuelvan automáticamente."""
    original = sample_dataset / "Ac" / "Ac-unique.png"
    duplicate = sample_dataset / "As" / "As-copy.png"
    _copy_exact_duplicate(original, duplicate)

    output = tmp_path / "data" / "curated"
    report_dir = tmp_path / "reports"

    summary = curate_dataset(
        input_dir=sample_dataset,
        output_dir=output,
        report_dir=report_dir,
    )

    # El archivo original de Ac y su copia en As se excluyen ambos, porque
    # el programa no tiene evidencia fiable para elegir la clase correcta.
    assert not (output / "Ac" / "Ac-unique.png").exists()
    assert not (output / "As" / "As-copy.png").exists()

    assert summary["images_kept"] == 2
    assert summary["images_excluded"] == 2
    assert summary["unresolved_cross_class_duplicate_groups"] == 1

    exclusions = _read_csv(report_dir / EXCLUSIONS_FILENAME)

    conflict_rows = [
        row
        for row in exclusions
        if row["reason"] == "cross_class_duplicate_label_unresolved"
    ]

    assert len(conflict_rows) == 2


def test_unreadable_images_are_excluded(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que los archivos de imagen corruptos se excluyan y documenten."""
    corrupt_path = sample_dataset / "Ac" / "corrupt.png"
    corrupt_path.write_bytes(b"not a real image")

    output = tmp_path / "data" / "curated"
    report_dir = tmp_path / "reports"

    summary = curate_dataset(
        input_dir=sample_dataset,
        output_dir=output,
        report_dir=report_dir,
    )

    assert not (output / "Ac" / "corrupt.png").exists()
    assert summary["images_excluded"] == 1

    exclusions = _read_csv(report_dir / EXCLUSIONS_FILENAME)

    corrupt_rows = [
        row
        for row in exclusions
        if row["relative_path"] == "Ac/corrupt.png"
    ]

    assert len(corrupt_rows) == 1
    assert corrupt_rows[0]["reason"] == "unreadable_or_hash_error"


def test_original_dataset_is_not_modified(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que los archivos de origen no cambien tras la curación."""
    source_files_before = sorted(
        path.relative_to(sample_dataset).as_posix()
        for path in sample_dataset.rglob("*")
        if path.is_file()
    )

    curate_dataset(
        input_dir=sample_dataset,
        output_dir=tmp_path / "data" / "curated",
        report_dir=tmp_path / "reports",
    )

    source_files_after = sorted(
        path.relative_to(sample_dataset).as_posix()
        for path in sample_dataset.rglob("*")
        if path.is_file()
    )

    assert source_files_after == source_files_before


def test_empty_class_directories_are_preserved(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que todas las clases originales permanezcan en el dataset curado."""
    # Se hace que la única imagen de Ac forme parte de un grupo de
    # duplicados entre clases.
    _copy_exact_duplicate(
        sample_dataset / "Ac" / "Ac-unique.png",
        sample_dataset / "As" / "As-copy.png",
    )

    output = tmp_path / "data" / "curated"

    curate_dataset(
        input_dir=sample_dataset,
        output_dir=output,
        report_dir=tmp_path / "reports",
    )

    assert (output / "Ac").is_dir()
    assert (output / "As").is_dir()
    assert (output / "Cu").is_dir()

    assert not any((output / "Ac").iterdir())


def test_audit_files_are_created(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que se escriban el manifiesto, las exclusiones y el resumen."""
    report_dir = tmp_path / "reports"

    curate_dataset(
        input_dir=sample_dataset,
        output_dir=tmp_path / "data" / "curated",
        report_dir=report_dir,
    )

    assert (report_dir / MANIFEST_FILENAME).is_file()
    assert (report_dir / EXCLUSIONS_FILENAME).is_file()
    assert (report_dir / SUMMARY_FILENAME).is_file()

    with (report_dir / SUMMARY_FILENAME).open(
        encoding="utf-8"
    ) as file:
        summary = json.load(file)

    assert summary["total_images_found"] == 3
    assert summary["images_kept"] == 3
    assert summary["images_excluded"] == 0


def test_manifest_records_keep_and_exclude_actions(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que cada imagen de origen reciba una acción registrada."""
    original = sample_dataset / "Ac" / "Ac-unique.png"
    _copy_exact_duplicate(original, sample_dataset / "Ac" / "Ac-copy.png")

    report_dir = tmp_path / "reports"

    curate_dataset(
        input_dir=sample_dataset,
        output_dir=tmp_path / "data" / "curated",
        report_dir=report_dir,
    )

    manifest = _read_csv(report_dir / MANIFEST_FILENAME)

    assert len(manifest) == 4
    assert all(row["action"] in {"keep", "exclude"} for row in manifest)

    kept = [row for row in manifest if row["action"] == "keep"]
    excluded = [row for row in manifest if row["action"] == "exclude"]

    assert len(kept) == 3
    assert len(excluded) == 1


def test_nonempty_destination_is_not_overwritten_by_default(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que los datos de salida existentes estén protegidos por defecto."""
    output = tmp_path / "data" / "curated"
    output.mkdir(parents=True)
    sentinel = output / "keep_me.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        curate_dataset(
            input_dir=sample_dataset,
            output_dir=output,
            report_dir=tmp_path / "reports",
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_overwrite_replaces_existing_destination(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que overwrite reemplace el destino cuando se solicita."""
    output = tmp_path / "data" / "curated"
    output.mkdir(parents=True)
    sentinel = output / "old_file.txt"
    sentinel.write_text("old output", encoding="utf-8")

    curate_dataset(
        input_dir=sample_dataset,
        output_dir=output,
        report_dir=tmp_path / "reports",
        overwrite=True,
    )

    assert not sentinel.exists()
    assert (output / "Ac" / "Ac-unique.png").is_file()


def test_source_and_destination_cannot_be_the_same(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que el dataset original no pueda usarse como salida."""
    with pytest.raises(ValueError, match="separate, non-nested"):
        curate_dataset(
            input_dir=sample_dataset,
            output_dir=sample_dataset,
            report_dir=tmp_path / "reports",
        )


def test_destination_cannot_be_inside_source(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que se rechace una salida anidada dentro del dataset de origen."""
    with pytest.raises(ValueError, match="separate, non-nested"):
        curate_dataset(
            input_dir=sample_dataset,
            output_dir=sample_dataset / "curated",
            report_dir=tmp_path / "reports",
        )


def test_missing_source_raises_error(tmp_path: Path) -> None:
    """Verifica que un origen inexistente lance FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        curate_dataset(
            input_dir=tmp_path / "missing",
            output_dir=tmp_path / "curated",
            report_dir=tmp_path / "reports",
        )


def test_report_directory_cannot_be_inside_curated_dataset(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que los reportes no puedan crear directorios de clase falsos."""
    output = tmp_path / "data" / "curated"

    with pytest.raises(ValueError, match="must not be inside"):
        curate_dataset(
            input_dir=sample_dataset,
            output_dir=output,
            report_dir=output / "reports",
        )