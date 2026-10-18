"""Pruebas unitarias del módulo de análisis exploratorio de imágenes de nubes.

Estas pruebas crean conjuntos de datos de imágenes temporales para poder
ejecutarse sin depender del dataset real de nubes.

Ejecutar desde la raíz del proyecto con:
    uv run pytest tests/test_eda.py -v
"""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from clouds_cnn.eda import (
    CLASS_BALANCE_FILENAME,
    DUPLICATES_FILENAME,
    IMAGE_PROPERTIES_FILENAME,
    INVENTORY_FILENAME,
    SUMMARY_FILENAME,
    find_exact_duplicates,
    inspect_images,
    run_eda,
    summarize_class_balance,
    summarize_image_properties,
)


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Crea un pequeño dataset artificial con tres clases de imágenes.

    El fixture crea:
        - Tres directorios de clase.
        - Dos imágenes legibles por clase.
        - Imágenes con distintas dimensiones.
        - Imágenes RGB que Pillow puede analizar.

    Args:
        tmp_path: Directorio temporal proporcionado por pytest.

    Returns:
        Ruta al directorio raíz del dataset artificial.
    """
    dataset_dir = tmp_path / "data" / "raw"

    # Cada clase tiene su propio directorio, como lo espera ImageFolder.
    class_names = ["Ac", "Cu", "St"]

    for class_index, class_name in enumerate(class_names):
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True)

        # Se crean dos imágenes por clase con propiedades ligeramente
        # distintas.
        for image_index in range(2):
            width = 32 + image_index * 8
            height = 24 + image_index * 4
            intensity = 40 + class_index * 60 + image_index * 10

            image = Image.new(
                "RGB",
                (width, height),
                color=(intensity, intensity, intensity),
            )

            image.save(class_dir / f"image_{image_index}.png")

    return dataset_dir


def test_inspect_images_finds_all_supported_images(
    sample_dataset: Path,
) -> None:
    """Verifica que el inventario contenga todas las imágenes soportadas."""
    records = inspect_images(sample_dataset)

    assert len(records) == 6
    assert all(record["readable"] for record in records)
    assert all(record["sha256"] for record in records)


def test_inspect_images_records_class_names(
    sample_dataset: Path,
) -> None:
    """Verifica que cada imagen se asigne a su clase (directorio padre)."""
    records = inspect_images(sample_dataset)

    discovered_classes = {record["class_name"] for record in records}

    assert discovered_classes == {"Ac", "Cu", "St"}


def test_inspect_images_records_dimensions_and_modes(
    sample_dataset: Path,
) -> None:
    """Verifica que las dimensiones y los modos de color se extraigan bien."""
    records = inspect_images(sample_dataset)

    assert all(record["width"] is not None for record in records)
    assert all(record["height"] is not None for record in records)
    assert all(record["mode"] == "RGB" for record in records)

    observed_dimensions = {
        (record["width"], record["height"])
        for record in records
    }

    assert (32, 24) in observed_dimensions
    assert (40, 28) in observed_dimensions


def test_inspect_images_calculates_brightness_and_contrast(
    sample_dataset: Path,
) -> None:
    """Verifica que las imágenes legibles tengan mediciones visuales numéricas."""
    records = inspect_images(sample_dataset)

    for record in records:
        assert record["brightness"] is not None
        assert record["contrast"] is not None
        assert isinstance(record["brightness"], float)
        assert isinstance(record["contrast"], float)


def test_inspect_images_ignores_unsupported_extensions(
    sample_dataset: Path,
) -> None:
    """Verifica que los archivos que no son imágenes se excluyan del inventario."""
    unsupported_file = sample_dataset / "Ac" / "notes.txt"
    unsupported_file.write_text("This is not an image.", encoding="utf-8")

    records = inspect_images(sample_dataset)

    assert len(records) == 6
    assert all(
        record["file_name"] != "notes.txt"
        for record in records
    )


def test_inspect_images_marks_corrupt_files_as_unreadable(
    sample_dataset: Path,
) -> None:
    """Verifica que una imagen corrupta se registre sin detener el escaneo."""
    corrupt_file = sample_dataset / "Ac" / "corrupt.png"
    corrupt_file.write_bytes(b"This is not a valid PNG image.")

    records = inspect_images(sample_dataset)

    corrupt_records = [
        record
        for record in records
        if record["file_name"] == "corrupt.png"
    ]

    assert len(corrupt_records) == 1
    assert corrupt_records[0]["readable"] is False
    assert corrupt_records[0]["error"] != ""


def test_inspect_images_raises_for_missing_directory(
    tmp_path: Path,
) -> None:
    """Verifica que un directorio de dataset inexistente lance FileNotFoundError."""
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        inspect_images(missing_dir)


def test_inspect_images_raises_when_no_class_directories_exist(
    tmp_path: Path,
) -> None:
    """Verifica que se rechace un dataset sin directorios de clase."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No class subdirectories"):
        inspect_images(empty_dir)


def test_summarize_class_balance_counts_each_class(
    sample_dataset: Path,
) -> None:
    """Verifica el conteo de imágenes y el porcentaje de cada clase."""
    records = inspect_images(sample_dataset)
    balance = summarize_class_balance(records, sample_dataset)

    assert len(balance) == 3
    assert all(row["image_count"] == 2 for row in balance)
    assert all(row["percentage"] == pytest.approx(33.3333) for row in balance)


def test_summarize_class_balance_includes_empty_classes(
    sample_dataset: Path,
) -> None:
    """Verifica que un directorio de clase vacío aparezca en el reporte de balance."""
    empty_class = sample_dataset / "Ns"
    empty_class.mkdir()

    records = inspect_images(sample_dataset)
    balance = summarize_class_balance(records, sample_dataset)

    empty_row = next(
        row for row in balance if row["class_name"] == "Ns"
    )

    assert empty_row["image_count"] == 0
    assert empty_row["percentage"] == 0.0


def test_summarize_image_properties_counts_readable_images(
    sample_dataset: Path,
) -> None:
    """Verifica los conteos de imágenes totales, legibles e ilegibles."""
    records = inspect_images(sample_dataset)
    summary = summarize_image_properties(records)

    assert summary["total_images"] == 6
    assert summary["readable_images"] == 6
    assert summary["unreadable_images"] == 0

def test_summarize_image_properties_handles_no_readable_images(
    tmp_path: Path,
) -> None:
    """Verifica que las estadísticas visuales sean nulas si ninguna imagen es legible."""
    dataset_dir = tmp_path / "data" / "raw"
    class_dir = dataset_dir / "Ac"
    class_dir.mkdir(parents=True)

    # Todas las imágenes del dataset son corruptas.
    (class_dir / "broken_0.png").write_bytes(b"invalid image")
    (class_dir / "broken_1.png").write_bytes(b"invalid image")

    records = inspect_images(dataset_dir)
    summary = summarize_image_properties(records)

    assert summary["total_images"] == 2
    assert summary["readable_images"] == 0
    assert summary["unreadable_images"] == 2

    # Sin imágenes legibles no hay dimensiones, modos ni estadísticas.
    assert summary["dimension_counts"] == {}
    assert summary["color_mode_counts"] == {}
    assert summary["brightness"]["mean"] is None
    assert summary["contrast"]["mean"] is None


def test_find_exact_duplicates_detects_identical_files(
    sample_dataset: Path,
) -> None:
    """Verifica que las copias idénticas byte a byte se detecten como duplicados."""
    original = sample_dataset / "Ac" / "image_0.png"
    duplicate = sample_dataset / "Cu" / "duplicate.png"

    # Se copian los bytes originales en lugar de guardar la imagen de nuevo.
    # Guardarla otra vez podría producir bytes distintos y, por tanto, un
    # hash diferente.
    shutil.copyfile(original, duplicate)

    records = inspect_images(sample_dataset)
    duplicates = find_exact_duplicates(records)

    assert len(duplicates) == 2

    duplicate_paths = {
        record["relative_path"]
        for record in duplicates
    }

    assert "Ac/image_0.png" in duplicate_paths
    assert "Cu/duplicate.png" in duplicate_paths


def test_find_exact_duplicates_returns_empty_list_without_duplicates(
    sample_dataset: Path,
) -> None:
    """Verifica que los archivos únicos no aparezcan en el reporte de duplicados."""
    records = inspect_images(sample_dataset)
    duplicates = find_exact_duplicates(records)

    assert duplicates == []


def test_run_eda_creates_expected_report_files(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que el flujo EDA escriba todos los reportes principales."""
    output_dir = tmp_path / "reports" / "eda"

    summary = run_eda(
        data_dir=sample_dataset,
        output_dir=output_dir,
        make_plots=False,
    )

    assert (output_dir / INVENTORY_FILENAME).is_file()
    assert (output_dir / CLASS_BALANCE_FILENAME).is_file()
    assert (output_dir / IMAGE_PROPERTIES_FILENAME).is_file()
    assert (output_dir / DUPLICATES_FILENAME).is_file()
    assert (output_dir / SUMMARY_FILENAME).is_file()

    assert summary["properties"]["total_images"] == 6


def test_run_eda_json_summary_is_valid(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que el JSON guardado pueda leerse y contenga los campos clave."""
    output_dir = tmp_path / "reports"

    run_eda(
        data_dir=sample_dataset,
        output_dir=output_dir,
        make_plots=False,
    )

    summary_path = output_dir / SUMMARY_FILENAME

    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)

    assert "properties" in summary
    assert "class_balance" in summary
    assert "balance_diagnostics" in summary
    assert "unreadable_images" in summary


def test_run_eda_reports_duplicate_groups(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que el resumen final reporte los grupos de duplicados exactos."""
    original = sample_dataset / "Ac" / "image_0.png"
    duplicate = sample_dataset / "St" / "same_image.png"
    shutil.copyfile(original, duplicate)

    summary = run_eda(
        data_dir=sample_dataset,
        output_dir=tmp_path / "reports",
        make_plots=False,
    )

    assert summary["exact_duplicate_groups"] == 1
    assert summary["images_in_exact_duplicate_groups"] == 2


def test_run_eda_reports_unreadable_images(
    sample_dataset: Path,
    tmp_path: Path,
) -> None:
    """Verifica que las imágenes corruptas aparezcan en el resumen final."""
    corrupt_file = sample_dataset / "St" / "corrupt.png"
    corrupt_file.write_bytes(b"not an image")

    summary = run_eda(
        data_dir=sample_dataset,
        output_dir=tmp_path / "reports",
        make_plots=False,
    )

    assert "St/corrupt.png" in summary["unreadable_images"]
    assert summary["properties"]["unreadable_images"] == 1