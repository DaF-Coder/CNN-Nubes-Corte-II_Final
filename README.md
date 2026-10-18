# CNN-MLOps — Clasificación de nubes CCSN

Proyecto académico basado en la arquitectura modular de **Galaxy-Morphology-k3-CNN-MLOps** del profesor. Se preservan sus responsabilidades (configuración, datos, CNN, motor, métricas, figuras, experimentos y MLflow) y se mantienen los módulos desarrollados para nubes: **EDA, curación y preparación desde carpetas locales**.

## Alcance y parámetros

- **11 clases:** `Ac`, `As`, `Cb`, `Cc`, `Ci`, `Cs`, `Ct`, `Cu`, `Ns`, `Sc`, `St` (orden alfabético de `ImageFolder`).
- **Imágenes:** RGB, `256 × 256` después de `Resize`. Las originales pueden tener otras dimensiones.
- **CNN:** cuatro bloques convolucionales, `AdaptiveAvgPool2d`, clasificador y 11 logits; arquitectura editable en `models.py`.
- **Entrenamiento:** `AdamW`, entropía cruzada, épocas fijas, checkpoint con menor pérdida de validación; sin scheduler ni early stopping, siguiendo al profesor.
- **Evaluación:** accuracy, precisión, recall, F1 por clase, macro F1, matriz `11 × 11` y generalization gap.

## Instalación (Windows PowerShell)

```powershell
uv sync
uv run pytest -q
uv run python -c "from clouds_cnn.config import ExperimentConfig; print(ExperimentConfig())"
```

`uv sync` usa `pyproject.toml` y `uv.lock`; Python 3.12. Si cambias dependencias, actualiza el lock con `uv lock`.

## Datos locales: no se descargan en el código

Coloca las imágenes originales en `data/raw/<clase>/`. El ZIP contiene las carpetas de las once clases, **pero no las fotografías reales**. Nunca se modifica `data/raw`.

```powershell
uv run python -m clouds_cnn.eda --data-dir data/raw --output-dir reports/eda
uv run python -m clouds_cnn.curate_dataset --input-dir data/raw --output-dir data/curated --report-dir reports/curation
uv run python -m clouds_cnn.preparation --source data/curated --output data/processed --image-size 256 --split-seed 2026
```

La preparación divide en `train/`, `val/`, `test/` (70/15/15), conserva los originales, escribe `manifest.csv` y calcula media/desviación **solo con train** en `dataset_stats.json`. La normalización de `data.py` usa esas estadísticas. Consulta `docs/DATOS_Y_CURACION.md`.

## Entrenamiento y evaluación

```powershell
uv run python -m clouds_cnn.train --run-name baseline --epochs 30
uv run python -m clouds_cnn.train --run-name batchnorm --epochs 30 --use-batchnorm
uv run python -m clouds_cnn.train --run-name batchnorm_dropout --epochs 30 --use-batchnorm --dropout-p 0.3
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Las tres configuraciones deben compararse con la **misma** partición y semilla. Elige usando solo validación y ejecuta `evaluate_on_test` **una sola vez**, tras seleccionar el modelo. El notebook `notebooks/02_experimentos_nubes.ipynb` muestra el flujo completo, métricas, curvas, matrices y mapas de características. **No hay métricas reales de CCSN precalculadas**.

## Organización

| Ruta | Responsabilidad | Qué modificar |
|---|---|---|
| `src/clouds_cnn/config.py` | Experimentos y semillas | épocas, batch, LR, semillas, dropout, BatchNorm |
| `src/clouds_cnn/models.py` | Arquitectura CNN | filtros, capas, unidades ocultas |
| `src/clouds_cnn/data.py` | Transformaciones y DataLoaders | transforms; normalización ligada a stats |
| `src/clouds_cnn/eda.py` | Diagnóstico de originales | informes, visualizaciones |
| `src/clouds_cnn/curate_dataset.py` | Duplicados y legibilidad | reglas de curación |
| `src/clouds_cnn/preparation.py` | Splits y estadísticas | proporciones, tamaño, semilla |
| `src/clouds_cnn/engine.py` | Train/validation | lógica del entrenamiento |
| `src/clouds_cnn/metrics.py` | Métricas | fórmulas y validaciones |
| `src/clouds_cnn/plots.py` | Figuras | presentación |
| `src/clouds_cnn/inspection.py` | Feature maps | capas observadas |
| `src/clouds_cnn/experiment.py` | Orquestación y MLflow | protocolo, registro y salidas |
| `src/clouds_cnn/train.py` | CLI | flags de ejecución |
| `tests/` | Pruebas unitarias e integración | tests con 11 clases y 256 px |
| `docs/` | Protocolo y resultados | decisiones y hallazgos reales |

**Importante:** los tests pequeños de 3 clases se conservan para verificar métricas manualmente; los tests específicos de CCSN comprueban el contrato de 11 clases. Los archivos de datos y resultados están ignorados por Git.

## Calidad y reproducibilidad

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

`docs/SPEC_EXPERIMENTO_1.md` describe el protocolo; `docs/PILOTO.md` y `docs/RESULTADOS.md` son plantillas que deben llenarse con **resultados reales**, nunca con las cifras de galaxias.
