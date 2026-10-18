# Especificación experimental — CNN para clasificación de 11 géneros de nubes

> **Principio rector:** **concepto → implementación → experimento → comparación**. Este documento amplía la especificación proporcionada, conserva el carácter de una CNN propia y distingue los cambios de arquitectura de los cambios de datos y de protocolo. No se presupone que ninguna modificación mejore el rendimiento.

## 1. Problema y pregunta de investigación

El objetivo es aprender una función de clasificación multiclase `f_θ : R^(3×256×256) → R^11` que asigne a cada fotografía un vector de **11 logits**. La clase predicha es `argmax_k f_θ(x)_k`. Los logits se utilizan en la pérdida de entropía cruzada, sin exigir una capa softmax explícita al final del modelo.

**Pregunta general:** ¿cómo cambia la capacidad de discriminación de una CNN propia sobre CCSN cuando se modifican la normalización, la regularización, la información de entrada y la escala espacial de las representaciones?

**Subpreguntas:** ¿el recorte inferior ayuda o elimina contexto útil? ¿BatchNorm y dropout mejoran la generalización? ¿kernels y campos receptivos mayores favorecen las nubes de organización extensa? ¿se distribuyen las mejoras de forma homogénea entre los once géneros?

## 2. Hipótesis y límites causales

- **H1 (regularización):** BatchNorm y dropout pueden modificar la estabilidad de entrenamiento y la brecha train–val; no garantizan una mejora de macro-F1.
- **H2 (entrada):** eliminar parte inferior de la fotografía puede reducir interferencias visuales, pero también retirar información meteorológica útil.
- **H3 (escala):** diferentes tamaños de kernel, operaciones de pooling y dilataciones modifican el campo receptivo y la representación espacial; podrían beneficiar a clases distintas.

Las hipótesis deben contrastarse con validación. Cuando varios factores cambian a la vez, el experimento permite comparar **configuraciones completas**, pero no atribuir causalmente el resultado a un único factor.

## 3. Dataset y contrato de partición

Se trabaja con CCSN local, inicialmente 2.543 fotografías en 11 clases, y con una copia curada de 2.523 tras excluir 20 duplicados exactos o conflictos. `data/raw/` no se modifica. El protocolo establece una partición estratificada 70/15/15 y semillas 2026 para split y entrenamiento. La normalización RGB se calcula exclusivamente con train. Las imágenes se redimensionan a 256×256 y se convierten a tensores.

En las evaluaciones observadas de 04–06 aparecen 1.766 ejemplos de train y 378 de val. `test` queda reservado para la fase final. La variación recortada del experimento 02 reside en un dataset derivado y requiere sus propias estadísticas de normalización.

## 4. Arquitectura original y significado de escala

La arquitectura original `CloudCNN` tiene cuatro bloques convolucionales con canales `(32, 64, 128, 256)`. Cada bloque aplica convolución, BatchNorm o identidad, ReLU y MaxPool 2×2. Tras ellos se utiliza `AdaptiveAvgPool2d((2,2))`, aplanamiento, capa oculta de 128 unidades, activación, dropout opcional y una capa final de 11 logits.

Las nuevas clases `CloudCNNPequena`, `CloudCNNMediana` y `CloudCNNGrande` se instancian directamente en los notebooks 04–06. En la ejecución de pequeña se observan convoluciones 3×3 y LeakyReLU(0.1); en mediana, la primera convolución es 5×5; en grande se modifica la escala espacial mediante kernels más amplios y dilatación según la propuesta arquitectónica del proyecto. Los módulos completos `models_pequena.py`, `models_mediana.py` y `models_grande.py` no fueron adjuntados en este lote: los detalles no visibles en el notebook no deben tratarse como una auditoría exhaustiva de su implementación.

### 4.1 Campo receptivo

El campo receptivo teórico es la región de entrada capaz de influir en una activación de una capa. Para una sucesión de capas, puede calcularse de forma recursiva:

\[
r_l=r_{l-1}+(k_l-1)d_lj_{l-1},\qquad j_l=j_{l-1}s_l,
\]

con `r_0=1`, `j_0=1`, tamaño de kernel `k_l`, dilatación `d_l` y stride `s_l`. Un kernel más grande o una dilatación mayor aumentan el campo receptivo teórico; el pooling también lo modifica al reducir la resolución espacial. **Campo receptivo grande no equivale automáticamente a mejor comprensión global**: influyen los datos, el entrenamiento y el campo receptivo efectivo.

### 4.2 Activación LeakyReLU

\[
\operatorname{LeakyReLU}_\alpha(x)=\begin{cases}x,&x\ge 0,\\\alpha x,&x<0.\end{cases}
\]

Con `α=0.1`, las activaciones negativas mantienen una pendiente distinta de cero, a diferencia de ReLU. Esto modifica el flujo local de gradientes, pero no demuestra que una arquitectura sea superior. En el estudio de escalas, LeakyReLU se mantiene común a las tres variantes para no confundir su efecto con el de los kernels.

### 4.3 MaxPool frente a AvgPool

MaxPool conserva el máximo de una ventana y enfatiza respuestas fuertes; AvgPool conserva su promedio y resume la actividad local. Ambos reducen resolución y pueden perder información. Alternarlos es una hipótesis arquitectónica sobre el equilibrio entre detalles salientes y patrones extendidos, no una garantía de que las categorías nubosas queden mejor separadas.

## 5. Matriz de las seis ejecuciones

| Nº | Notebook | Run | Intervención | Épocas | Batch | BN | Dropout |
|:--:|:--|:--|:--|--:|--:|:--:|--:|
| 01 | `01_experimentos_nubes_basline.ipynb` | `baseline` | CNN original | 30 | 32 | No | 0 |
| 02 | `02_experimento_recote(2).ipynb` | `recorte_inferior` | Recorte inferior + CNN original | 15 | 64 | Sí | Ver configuración efectiva |
| 03 | `03_experimento_dropout_03.ipynb` | `droput_03` | CNN original + BN + dropout | 15 | 64 | Sí | 0.3 |
| 04 | `04_experimento_escala_pequena(1).ipynb` | `escala_pequena_04` | LeakyReLU, escala pequeña | 15 | 64 | Sí | 0.3 |
| 05 | `05_experimento_escala_mediana(1).ipynb` | `escala_mediana_05` | LeakyReLU, escala mediana | 15 | 64 | Sí | **0.4** |
| 06 | `06_experimento_escala_grande(1).ipynb` | `escala_grande_06` | LeakyReLU, escala grande | 15 | 64 | Sí | 0.3 |

**Advertencias de trazabilidad:** el nombre literal `droput_03` conserva la grafía del notebook. La versión inicial de los cuadernos 05/06 tenía el sufijo `_04`; las versiones adjuntas de 05/06 utilizan `_05` y `_06`. Debe verificarse que cada checkpoint corresponda a la arquitectura y al run declarados.

## 6. Optimización, checkpoint y recuperación

La pérdida de entrenamiento es la entropía cruzada multiclase:

\[
\mathcal L(\theta)=-\frac1B\sum_{i=1}^B\log\frac{\exp(z_{i,y_i})}{\sum_{k=1}^{11}\exp(z_{i,k})}.
\]

Se emplea AdamW, learning rate inicial `0.001` y `weight_decay=0` según las configuraciones observadas. La elección del checkpoint se hace por **mínimo `val_loss`**. En 04–06 el entrenamiento se ejecuta con `fit()` de `clouds_cnn.engine` y el modelo concreto se instancia desde su módulo (`models_pequena`, `models_mediana` o `models_grande`), porque `run_experiment()` de la CNN original no selecciona automáticamente esas clases.

La ruta del checkpoint sigue `outputs/checkpoints/<run_name>_best.pt`. La existencia del archivo no prueba que finalizaran las épocas; puede haberse guardado durante un entrenamiento interrumpido. Antes de recuperar se debe verificar nombre de experimento, clase del modelo, compatibilidad de pesos y finalización de la ejecución. Reentrenar con sobrescritura requiere una decisión explícita; no debe mezclarse con consultar métricas.

## 7. Métricas y fórmulas

Sea `M` la matriz de confusión, con **filas = clases reales** y **columnas = clases predichas**. Para la clase `i`:

\[
TP_i=M_{ii},\quad N_i=\sum_jM_{ij},\quad P_i=\sum_jM_{ji}.
\]

\[
\mathrm{recall}_i=\frac{TP_i}{N_i},\qquad
\mathrm{precision}_i=\frac{TP_i}{P_i},\qquad
F1_i=\frac{2\,\mathrm{precision}_i\,\mathrm{recall}_i}{\mathrm{precision}_i+\mathrm{recall}_i}.
\]

\[
\mathrm{accuracy}=\frac{\sum_iM_{ii}}{\sum_{i,j}M_{ij}},\quad
\mathrm{macroF1}=\frac1{11}\sum_iF1_i,\quad
\mathrm{weightedF1}=\frac{\sum_iN_iF1_i}{\sum_iN_i}.
\]

En los notebooks, `accuracy_clase = aciertos_i / soporte_i` es **idéntico a recall_i**. No debe interpretarse como una métrica adicional. Cuando una clase carece de soporte, la división debe marcarse como indefinida (`NaN`) o seguir una política explícita.

### 7.1 Incertidumbre

Los intervalos de Wilson aproximan la incertidumbre de una proporción `p=x/n`:

\[
\frac{p+z^2/(2n)\pm z\sqrt{p(1-p)/n+z^2/(4n^2)}}{1+z^2/n},\qquad z=1.96.
\]

Son útiles para recordar que un recall de 70 % sobre 30 imágenes es menos preciso que sobre 3.000. Su interpretación supone observaciones aproximadamente independientes; fotografías relacionadas o selección reiterada sobre validación pueden reducir la validez de esa aproximación.

## 8. Protocolo de comparación

1. Ejecutar EDA y curación sin alterar `raw`.
2. Fijar manifiesto y normalización antes de comparar modelos.
3. Verificar clase del modelo, parámetros, ruta y `run_name` en cada notebook.
4. Entrenar durante el presupuesto declarado y registrar las curvas completas.
5. Recuperar **el mejor checkpoint por val_loss** y evaluar train/val en modo `eval()`.
6. Reportar soporte, matriz de confusión, accuracy, macro-F1 y F1 ponderado.
7. Comparar diferencias por clase, no solo el promedio global.
8. Elegir la configuración sin consultar test.
9. Ejecutar la evaluación final de test **una sola vez**, con el modelo seleccionado.

Para aislar la escala espacial, repetir 05 con dropout 0.3 y mantener constantes las demás variables. La comparación actual 04–06 es **observacional entre configuraciones no completamente controladas**. Varias semillas permitirían estimar variabilidad de entrenamiento; los notebooks actuales no proporcionan réplicas independientes.

## 9. Entregables y estructura esperada

- `data/processed/manifest.csv`, `dataset_stats.json` y directorios `train/val/test`.
- `outputs/checkpoints/<run_name>_best.pt` por experimento.
- Historial de `train_loss`, `val_loss`, `train_acc`, `val_acc`, tiempo y mejor época.
- Tablas CSV de métricas globales y por clase; matrices de confusión absolutas y normalizadas.
- Gráficas de curvas, balance, matrices de confusión y mapas de características.
- Registro de hardware, semillas, dependencias y versión de arquitectura.
- Documento final que distinga observaciones, interpretaciones y limitaciones.

## 10. Exclusiones y alcance

No se incorporan ResNet, modelos preentrenados, scheduler, early stopping, DDP, servicios externos ni búsqueda masiva de hiperparámetros. El objetivo es estudiar arquitecturas propias con una secuencia explicable y verificable. Tampoco se deben extrapolar los resultados de validación a test sin evaluarlo.

## 11. Fuentes internas

`SPEC_EXPERIMENTO_1.md` original y notebooks 01–06 proporcionados. Los detalles del módulo de cada escala que no aparecen impresos en los notebooks requieren inspección de los archivos `.py` originales para una auditoría exhaustiva.
