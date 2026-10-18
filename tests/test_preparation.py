
def test_prepare_defaults_match_experiment_config() -> None:
    """Comprueba que la preparación use los valores predeterminados del experimento."""
    from inspect import signature

    from clouds_cnn.config import ExperimentConfig
    from clouds_cnn.preparation import prepare

    parameters = signature(prepare).parameters
    config = ExperimentConfig()

    assert parameters["split_seed"].default == config.split_seed
    assert parameters["image_size"].default == config.image_size


def test_prepare_records_all_splits_and_class_labels(
    prepared_dataset,
) -> None:
    """Comprueba que los metadatos registren los splits y las tres clases."""
    import json

    from clouds_cnn.preparation import MANIFEST_NAME, SPLITS, STATS_NAME

    manifest_path = prepared_dataset / MANIFEST_NAME
    stats_path = prepared_dataset / STATS_NAME

    assert manifest_path.is_file()
    assert stats_path.is_file()

    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    assert set(stats["class_to_label"]) == {"Ac", "As", "Cb"}
    assert set(SPLITS) == {"train", "val", "test"}
    assert stats["split_seed"] == 2026
    assert stats["transform"]["image_size"] == 16

