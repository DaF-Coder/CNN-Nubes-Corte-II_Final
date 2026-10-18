# Resultados experimentales — seis CNN para CCSN

> **Origen y alcance de la evidencia.** Este documento se construyó leyendo las **salidas ejecutadas guardadas dentro de los notebooks adjuntos**, no ejecutando nuevos entrenamientos ni consultando los archivos `outputs/` del ordenador. Las cifras de 04–06 provienen de sus evaluaciones recuperadas con el mejor checkpoint. El notebook 02 no conserva una evaluación ejecutada suficiente para asignarle resultados. No se dispone aquí de una evaluación final comparable de test para las seis arquitecturas.

## 1. Pregunta y criterio de evaluación

El problema es clasificar 11 géneros de nubes con CNN propias, modificando gradualmente datos, normalización, dropout y escala de representación. El criterio de **checkpoint** es mínimo `val_loss`; el criterio especificado para **comparar configuraciones** es macro-F1 de validación. Accuracy, pérdida, F1 ponderado, brecha train–val y confusiones por clase complementan esa comparación.

## 2. Tabla comparativa de resultados observados

| Exp. | Run | Épocas | Dropout | Mejor época | Val loss | Accuracy train | Accuracy val | Macro-F1 val | Brecha (pp) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 01 | baseline | 30 | 0 | 19 | 1.7990 | 52.66 % | 37.04 % | No comprobado | 15.62 |
| 02 | recorte_inferior | 15 | Según notebook | — | — | — | — | — | — |
| 03 | droput_03 | 15 | 0.3 | 12 | 1.7565 | 45.70 % | 37.57 % | 0.3192 | 8.13 |
| 04 | escala_pequena_04 | 15 | 0.3 | 12 | **1.7346** | 47.73 % | **38.89 %** | **0.3509** | 8.84 |
| 05 | escala_mediana_05 | 15 | **0.4** | 12 | 1.7789 | 44.62 % | 36.24 % | 0.3195 | 8.38 |
| 06 | escala_grande_06 | 15 | 0.3 | 12 | 1.8844 | 40.77 % | 29.63 % | 0.2384 | 11.14 |

Las cifras de accuracy train son evaluaciones del checkpoint seleccionado, no necesariamente el accuracy de entrenamiento mostrado en la última época. Los 04–06 usan **1.766** imágenes train y **378** imágenes val. Para 03, la cifra de train se toma de `final_metrics()` del notebook. Las comparaciones con 01 no aíslan un único cambio, porque 01 utiliza 30 épocas y batch size 32, mientras que los otros utilizan 15 y batch size 64.

## 3. Resultado central: la escala pequeña obtiene el mejor rendimiento observado entre 04–06

En la validación de 378 imágenes, 04 acierta **147**, 05 acierta **137** y 06 acierta **112** (el último conteo se deduce del 29.63 % de accuracy sobre 378 imágenes). Por tanto:

- **04 frente a 05:** +2.65 puntos porcentuales de accuracy y +0.0314 de macro-F1.
- **04 frente a 06:** +9.26 puntos porcentuales de accuracy y +0.1125 de macro-F1.
- **05 frente a 06:** +6.61 puntos porcentuales de accuracy y +0.0811 de macro-F1.

Estas diferencias son **descriptivas**, no pruebas de significación estadística ni estimaciones de mejora causal. Los tres modelos usan la misma partición de validación, pero no se dispone aquí de una comparación pareada por imagen ni de réplicas con varias semillas. Además, 05 tiene dropout 0.4: por ello no puede atribuirse la diferencia 04–05 exclusivamente al tamaño de los kernels o al pooling.

## 4. Rendimiento por clase: tablas de validación

Las tablas siguientes proceden de las salidas HTML embebidas en los notebooks. `accuracy_clase` se interpreta como **aciertos/soporte**, equivalente a recall; se conserva junto a recall por fidelidad al reporte, pero no aporta información independiente.

### 4.1 Experimento 04 — escala pequeña

| clase   |   soporte |   aciertos | accuracy_clase   | precision   | recall   | f1     |
|:--------|----------:|-----------:|:-----------------|:------------|:---------|:-------|
| Ac      |        33 |          5 | 15.15%           | 22.73%      | 15.15%   | 18.18% |
| As      |        28 |          6 | 21.43%           | 37.50%      | 21.43%   | 27.27% |
| Cb      |        35 |         19 | 54.29%           | 38.00%      | 54.29%   | 44.71% |
| Cc      |        39 |         18 | 46.15%           | 50.00%      | 46.15%   | 48.00% |
| Ci      |        21 |          3 | 14.29%           | 15.79%      | 14.29%   | 15.00% |
| Cs      |        43 |         11 | 25.58%           | 28.95%      | 25.58%   | 27.16% |
| Ct      |        30 |         21 | 70.00%           | 61.76%      | 70.00%   | 65.62% |
| Cu      |        27 |          8 | 29.63%           | 57.14%      | 29.63%   | 39.02% |
| Ns      |        41 |         19 | 46.34%           | 34.55%      | 46.34%   | 39.58% |
| Sc      |        51 |         35 | 68.63%           | 38.04%      | 68.63%   | 48.95% |
| St      |        30 |          2 | 6.67%            | 100.00%     | 6.67%    | 12.50% |

La distribución es heterogénea: `Ct` tiene recall de 70 % y `Sc` de 68.63 %, mientras que `St` solo llega a 6.67 % (2 de 30 imágenes). Para `St`, la precisión del 100 % no es una contradicción: el modelo predice esa clase únicamente en dos casos y ambos son correctos; sin embargo, omite 28 imágenes reales de esa clase. Una precisión alta con recall bajo puede resultar de un clasificador muy restrictivo.

### 4.2 Experimento 05 — escala mediana

| clase   |   soporte |   aciertos | accuracy_clase   | precision   | recall   | f1     |
|:--------|----------:|-----------:|:-----------------|:------------|:---------|:-------|
| Ac      |        33 |          5 | 15.15%           | 25.00%      | 15.15%   | 18.87% |
| As      |        28 |         10 | 35.71%           | 43.48%      | 35.71%   | 39.22% |
| Cb      |        35 |         19 | 54.29%           | 34.55%      | 54.29%   | 42.22% |
| Cc      |        39 |         16 | 41.03%           | 55.17%      | 41.03%   | 47.06% |
| Ci      |        21 |          3 | 14.29%           | 23.08%      | 14.29%   | 17.65% |
| Cs      |        43 |         10 | 23.26%           | 28.57%      | 23.26%   | 25.64% |
| Ct      |        30 |         21 | 70.00%           | 63.64%      | 70.00%   | 66.67% |
| Cu      |        27 |          2 | 7.41%            | 40.00%      | 7.41%    | 12.50% |
| Ns      |        41 |         15 | 36.59%           | 27.78%      | 36.59%   | 31.58% |
| Sc      |        51 |         35 | 68.63%           | 32.11%      | 68.63%   | 43.75% |
| St      |        30 |          1 | 3.33%            | 50.00%      | 3.33%    | 6.25%  |

La caída de macro-F1 respecto a 04 no implica que todas las clases empeoren. Es necesario inspeccionar los cambios por clase y las confusiones específicas. Dado el dropout 0.4, la diferencia puede deberse a la arquitectura, la regularización o su interacción.

### 4.3 Experimento 06 — escala grande

| clase   |   soporte |   aciertos | accuracy_clase   | precision   | recall   | f1     |
|:--------|----------:|-----------:|:-----------------|:------------|:---------|:-------|
| Ac      |        33 |          4 | 12.12%           | 22.22%      | 12.12%   | 15.69% |
| As      |        28 |          4 | 14.29%           | 28.57%      | 14.29%   | 19.05% |
| Cb      |        35 |         18 | 51.43%           | 25.71%      | 51.43%   | 34.29% |
| Cc      |        39 |         17 | 43.59%           | 25.00%      | 43.59%   | 31.78% |
| Ci      |        21 |          1 | 4.76%            | 20.00%      | 4.76%    | 7.69%  |
| Cs      |        43 |          3 | 6.98%            | 23.08%      | 6.98%    | 10.71% |
| Ct      |        30 |         16 | 53.33%           | 69.57%      | 53.33%   | 60.38% |
| Cu      |        27 |          1 | 3.70%            | 8.33%       | 3.70%    | 5.13%  |
| Ns      |        41 |         19 | 46.34%           | 28.79%      | 46.34%   | 35.51% |
| Sc      |        51 |         29 | 56.86%           | 33.33%      | 56.86%   | 42.03% |
| St      |        30 |          0 | 0.00%            | 0.00%       | 0.00%    | 0.00%  |

El macro-F1 de 0.2384 y accuracy de 29.63 % muestran una pérdida general de capacidad discriminativa respecto a 04 y 05 bajo este entrenamiento. No significa que un campo receptivo grande sea intrínsecamente peor para las nubes: puede haber interacciones con el número de épocas, la forma de reducción espacial, la capacidad efectiva de aprendizaje y la optimización.


## 5. Balance de clases, soporte e incertidumbre

El soporte de validación varía entre clases (por ejemplo, 21 para `Ci`, 30 para `St` y 51 para `Sc`). Esto afecta la estabilidad de las métricas: un acierto adicional sobre 21 imágenes altera el recall en unos 4.76 puntos porcentuales, mientras que sobre 51 lo modifica en unos 1.96 puntos. Por ello los notebooks incorporan intervalos de Wilson del 95 %.

Un intervalo de Wilson no corrige automáticamente el sesgo de selección por validación repetida, ni garantiza independencia entre imágenes de una misma fuente. El conjunto de validación se utilizó para elegir checkpoints; sus métricas son estimaciones de desarrollo, no una medición final sobre datos nunca consultados.

## 6. Evolución durante las épocas y selección del checkpoint

Los experimentos 03–06 alcanzan el mínimo de `val_loss` en la época **12**, aunque continúan hasta la época 15. La selección del checkpoint preserva el mejor estado de validación en vez de utilizar necesariamente los últimos pesos. En 04, por ejemplo, la última época tiene accuracy de validación de alrededor de 34.13 %, mientras que el checkpoint de la época 12 obtiene 38.89 %. Esto ilustra por qué no deben confundirse **última época** y **mejor época**.

Las curvas muestran una reducción general de la pérdida de entrenamiento, pero el comportamiento de validación es irregular. No basta con afirmar sobreajuste únicamente por la brecha positiva: también hay rendimientos de entrenamiento modestos, lo que exige considerar aprendizaje insuficiente, representación y regularización.

## 7. Validez de la comparación y factores de confusión

**Factor 1 — Dropout distinto en 05.** El notebook adjunto fija `dropout_p=0.4`, frente a 0.3 en 04 y 06. Repetir 05 con 0.3 es el control prioritario.

**Factor 2 — Cambios arquitectónicos simultáneos.** Tamaños de kernel, pooling y dilatación pueden variar juntos; el resultado identifica el desempeño de la configuración, no el efecto aislado de una operación.

**Factor 3 — Diferente presupuesto del baseline.** El 01 usa 30 épocas y batch 32, por lo que no es un control idéntico de 04–06.

**Factor 4 — Una sola semilla.** `train_seed=2026` aparece en la configuración, pero no se dispone de repeticiones independientes. `deterministic=False` también aparece en los objetos de configuración, lo que limita la garantía de reproducibilidad bit a bit.

**Factor 5 — Selección en validación.** Elegir la mejor época y comparar modelos con el mismo val puede producir una estimación optimista respecto al test final.

**Factor 6 — Trazabilidad.** Versiones anteriores de los cuadernos 05 y 06 tenían nombres `_04`. Los adjuntos actuales declaran `_05` y `_06`, pero la existencia de checkpoints antiguos exige comprobar metadatos antes de asociar un resultado con una arquitectura.

## 8. Conclusiones sustentadas y conclusiones que no se pueden afirmar

**Sustentado por los notebooks:** entre las ejecuciones 04, 05 y 06 presentadas, la pequeña obtuvo el mayor accuracy, macro-F1 y la menor pérdida de validación. El rendimiento de las tres varía mucho entre clases. El experimento 06 no mejoró los resultados de 04 bajo el protocolo observado.

**Evidencia Insuficiente:** que la escala pequeña sea universalmente superior, que los kernels grandes sean perjudiciales, que la causa de los errores sea únicamente el campo receptivo, que 05 sea peor exclusivamente por su escala o que estos porcentajes describan el rendimiento final sobre test.

## 9. Siguientes experimentos recomendados

1. Repetir 05 con dropout 0.3, manteniendo el resto de la configuración.
2. Registrar los tres modelos con una tabla verificable de kernels, pooling, dilatación, parámetros y campo receptivo calculado.
3. Comparar errores **pareados por imagen** y clases especialmente difíciles (`St`, `Ci`, `Ac`).
4. Repetir 04–06 con varias semillas y reportar media y dispersión de macro-F1.
5. Conservar la elección de modelo por macro-F1 de validación; evaluar test únicamente después de fijar la selección.
6. Documentar el hardware y el tiempo por época, pues kernels más grandes pueden cambiar el coste computacional sin aportar mejora de validación.


