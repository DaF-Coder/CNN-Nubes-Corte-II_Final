## 1. Función del piloto

El piloto permite verificar que el pipeline de datos, el entrenamiento y la evaluación funcionan de extremo a extremo antes de extraer conclusiones sobre arquitecturas. Sus objetivos son: (a) establecer un rendimiento inicial; (b) medir el comportamiento de las curvas de aprendizaje; (c) comprobar tiempos y recursos; (d) fijar una política de selección del mejor checkpoint; y (e) definir qué se mantiene constante al comparar modelos.

Una CNN puede disminuir la pérdida de entrenamiento sin mejorar validación. Por eso el piloto no debe basarse únicamente en el último valor de `train_acc`, sino examinar `val_loss`, `val_acc`, macro-F1 y matrices de confusión.

## 2. Configuración provisional del documento original

El documento de partida propuso `image_size=256`, `num_classes=11`, `center_crop=None`, `epochs=30`, `split_seed=2026`, `train_seed=2026`, `batch_size=32`, `learning_rate=1e-3` y `weight_decay=0`. Esos valores son el punto de partida del baseline, **no la configuración uniforme de las seis ejecuciones**.

En los notebooks 02–06 se emplean **15 épocas y batch size 64**. El notebook 05 emplea **dropout 0.4**, mientras que los notebooks 03, 04 y 06 emplean **0.3**. La diferencia debe documentarse: invalida una interpretación de la comparación 04–06 como efecto *exclusivo* de la escala de representación.

## 3. Secuencia efectiva de experimentos

| Experimento | Intervención principal | Datos | Épocas | Batch | BatchNorm | Dropout |
|:--|:--|:--|--:|--:|:--:|--:|
| 01 baseline | CNN original, sin BN ni dropout | `processed` | 30 | 32 | No | 0 |
| 02 recorte | CNN original, recorte inferior | dataset recortado | 15 | 64 | Sí | 0 según especificación; verificar configuración final |
| 03 dropout | CNN original con BN y dropout | `processed` | 15 | 64 | Sí | 0.3 |
| 04 pequeña | `CloudCNNPequena` | `processed` | 15 | 64 | Sí | 0.3 |
| 05 mediana | `CloudCNNMediana` | `processed` | 15 | 64 | Sí | **0.4** |
| 06 grande | `CloudCNNGrande` | `processed` | 15 | 64 | Sí | 0.3 |

El baseline de 30 épocas no es directamente comparable, como intervención aislada, con los de 15: además de la arquitectura y la regularización, cambia el presupuesto de optimización. El recorte cambia los datos de entrada, por lo que requiere interpretación separada.

## 4. Procedimiento de entrenamiento

El flujo parte de `ExperimentConfig` y de un dataset ya preparado. En 01–03 se utiliza la infraestructura de experimentos de la CNN original; en 04–06 se construyen directamente clases distintas y se llama a `clouds_cnn.engine.fit()` para evitar que `run_experiment()` instancie inadvertidamente el modelo original.

En cada época, la red calcula logits para cada lote, minimiza entropía cruzada y ajusta parámetros con AdamW y retropropagación. El *checkpoint* elegido corresponde al **mínimo `val_loss`**, no necesariamente a la época con mayor `val_acc`. Esta distinción es sustantiva: la entropía cruzada considera las probabilidades asignadas a las etiquetas correctas, mientras que el accuracy solo considera si la clase más probable coincide con la etiqueta.

## 5. Resultados de piloto observables

| Experimento | Mejor época | Mejor val loss | Accuracy train del checkpoint | Accuracy val | Macro-F1 val |
|:--|--:|--:|--:|--:|--:|
| 01 baseline | 19 | 1.7990 | 52.66 % | 37.04 % | 0.3274 |
| 02 recorte | 14  | 1.8034 | 45.30 % | 36.77 % | 0.3368 |
| 03 dropout | 12 | 1.7565 | 45.70 % | 37.57 % | 0.3192 |
| 04 pequeña | 12 | 1.7346 | 47.73 % | 38.89 % | 0.3509 |
| 05 mediana | 12 | 1.7789 | 44.62 % | 36.24 % | 0.3195 |
| 06 grande | 12 | 1.8844 | 40.77 % | 29.63 % | 0.2384 |

**Fuente:** salidas ejecutadas y conservadas dentro de los notebooks adjuntos. Para 01 el resumen del cuaderno indica `train_accuracy=0.5266` y `val_accuracy=0.3704`; para 03 se conserva un resumen explícito de `final_metrics()`. Para 04–06 se conservan las evaluaciones de train y val con el checkpoint recuperado.

La mejor época de los experimentos 03–06 fue 12, pero **eso no demuestra** que 12 sea el número óptimo universal de épocas. En los cuadernos se siguió entrenando hasta 15 y se eligió el mejor estado por pérdida de validación.

## 6. Curvas y brecha de generalización

La brecha se define como `accuracy_train_eval − accuracy_val`, con ambas métricas calculadas usando el mismo checkpoint y el modelo en modo evaluación. No debe confundirse con restar el accuracy de train de una época al de validación si los estados o las condiciones de evaluación son diferentes.

| Experimento | Brecha (puntos porcentuales, aprox.) |
|:--|--:|
| 01 | 15.62 |
| 03 | 8.13 |
| 04 | 8.84 |
| 05 | 8.38 |
| 06 | 11.14 |

Una brecha positiva sugiere peor rendimiento fuera de entrenamiento, pero no basta para diagnosticar sobreajuste: un modelo con accuracy de entrenamiento bajo también puede presentar subajuste o capacidad de representación insuficiente. En los experimentos 04–06, los valores de train no son altos, por lo que conviene evitar conclusiones simplistas.

## 7. Decisiones metodológicas y correcciones necesarias

**Decisión 1 — Métrica de selección.** Mantener `val_loss` para seleccionar checkpoint y **macro-F1 de validación** para comparar arquitecturas, tal como indica la especificación. El accuracy global es complementario.

**Decisión 2 — Control de confusiones.** Analizar la matriz de confusión con filas reales y columnas predichas. En la documentación por clase, la razón `aciertos/soporte` es el **recall de la clase**, aunque en algunos notebooks se denomine `accuracy_clase`. No es una métrica independiente del recall.

**Decisión 3 — Etiquetado de experimentos.** Usar `escala_pequena_04`, `escala_mediana_05` y `escala_grande_06`. Los cuadernos anteriores tenían sufijos `_04` incorrectos para 05 y 06. El nombre del archivo `.ipynb` no corrige automáticamente el `run_name` ni la ruta del checkpoint.

**Decisión 4 — Checkpoints incompletos.** La existencia de un archivo `*_best.pt` demuestra que se guardó algún estado, no que finalizaron todas las épocas. La recuperación y la evaluación deben comprobar la identidad de arquitectura y la integridad del registro de ejecución.

**Decisión 5 — Comparación controlada.** Para una comparación de escalas, repetir 05 con dropout 0.3, mantener idénticos datos, semillas, batch, épocas, optimizador y normalización, y documentar las diferencias arquitectónicas. Idealmente, repetir cada configuración con varias semillas.

**Decisión 6 — Test.** Mantenerlo reservado hasta la selección definitiva. El hecho de visualizar conteos de test no implica usar sus etiquetas para escoger un modelo, pero su matriz de confusión o accuracy no deben intervenir en el piloto.

## 8. Registro experimental ampliado

| Campo | Registro observado o pendiente |
|:--|:--|
| Dataset | CCSN curado; 2.523 imágenes |
| Clases | 11 |
| Resolución | 256×256 RGB |
| Semillas declaradas | split 2026; train 2026 |
| Dispositivo 04–06 | CPU, según salidas de notebooks |
| Versión de PyTorch 04–06 | `2.14.1+cpu`, según impresión de notebook |
| Tiempo por época | Aproximadamente 160–205 s en 04–06; varía según época |
| Identificador MLflow | No comprobado en las salidas adjuntas |
| Fecha exacta de cada entrenamiento | No comprobada |
| Test final | No se presenta como comparación de las seis ejecuciones |

