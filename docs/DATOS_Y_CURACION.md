# Datos locales, EDA, curación y preparación del conjunto CCSN

> **Alcance y trazabilidad.** Documento metodológico ampliado de `DATOS_Y_CURACION.md`, elaborado a partir de los cuadernos 01–06 adjuntos. Los números que se indican como *observados* provienen de las salidas conservadas en esos cuadernos; las explicaciones conceptuales se distinguen de esas observaciones. No se ha accedido al disco local de Windows ni a los archivos originales de las fotografías.

## 1. Objetivo y criterio de conservación

El objetivo es construir un conjunto de datos reproducible para clasificar fotografías de nubes en **11 géneros** mediante una CNN desarrollada en PyTorch. El procedimiento conserva el principio de **no descargar fotografías desde el código** y de **no modificar los originales**. Las carpetas `data/raw/<clase>/` constituyen la fuente primaria; la curación se efectúa sobre copias. Esto evita que una ejecución posterior altere silenciosamente el material del que se obtuvieron los resultados.

El trabajo diferencia cuatro objetos: (i) imágenes originales, (ii) copia curada, (iii) particiones estratificadas y estadísticas de entrenamiento y (iv) variantes experimentales, como el recorte inferior. Es esencial no confundir estas etapas: el modelo puede cambiar sin que cambie el dataset, mientras que el experimento 02 sí cambia la información visual disponible.

## 2. Identificación de clases y distribución original

El dataset CCSN empleado contiene **2.543 imágenes originales**, agrupadas en las etiquetas `Ac`, `As`, `Cb`, `Cc`, `Ci`, `Cs`, `Ct`, `Cu`, `Ns`, `Sc` y `St`. El inventario inicial registrado por el notebook 01 fue:

| Clase | Imágenes originales |
|:--|--:|
| Ac | 221 |
| As | 188 |
| Cb | 242 |
| Cc | 268 |
| Ci | 139 |
| Cs | 287 |
| Ct | 200 |
| Cu | 182 |
| Ns | 274 |
| Sc | 340 |
| St | 202 |
| **Total** | **2.543** |

La distribución no es uniforme: `Sc` tiene más del doble de ejemplos que `Ci`. Esta diferencia debe considerarse en la lectura del accuracy global: una clase frecuente puede contribuir más a la exactitud total que una minoritaria. Por ello se reportan soporte, precisión, recall, F1 por clase y macro-F1.

## 3. Análisis exploratorio (EDA)

El notebook 01 llama a `clouds_cnn.eda` antes y después de curar. La inspección abarca el inventario de archivos, el balance de clases, propiedades de imagen, brillo/contraste, ejemplos visuales y duplicados exactos. Entre las salidas registradas aparecen `image_inventory.csv`, `image_properties.csv`, `class_balance.csv`, `class_balance.png`, `brightness_contrast.png`, `class_examples.png`, `exact_duplicates.csv` y `eda_summary.json`.

**Justificación.** Una CNN no observa etiquetas semánticas, sino tensores de intensidades. Diferencias de resolución, iluminación, compresión, fondo o proporción de clases pueden influir en la optimización y en la generalización. El EDA no prueba por sí mismo que haya sesgos; permite identificarlos antes de atribuir el rendimiento a la arquitectura.

## 4. Curación: duplicados y conflictos de etiqueta

La ejecución registrada de `clouds_cnn.curate_dataset` encontró **2.543** archivos y conservó **2.523**: excluyó **20**, de los cuales **14** eran duplicados redundantes dentro de una misma clase y **6** pertenecían a grupos de duplicados exactos con etiquetas cruzadas sin resolver (tres grupos conflictivos). La curación generó una copia en `data/curated/` y mantuvo los originales intactos.

| Clase | Originales | Curadas | Excluidas |
|:--|--:|--:|--:|
| Ac | 221 | 219 | 2 |
| As | 188 | 185 | 3 |
| Cb | 242 | 236 | 6 |
| Cc | 268 | 265 | 3 |
| Ci | 139 | 139 | 0 |
| Cs | 287 | 286 | 1 |
| Ct | 200 | 200 | 0 |
| Cu | 182 | 177 | 5 |
| Ns | 274 | 274 | 0 |
| Sc | 340 | 340 | 0 |
| St | 202 | 202 | 0 |
| **Total** | **2.543** | **2.523** | **20** |

**Razón estadística.** Un mismo archivo repetido en entrenamiento y validación podría permitir reconocer la fotografía, en vez de aprender características generalizables de las nubes. Los conflictos de etiqueta, por su parte, hacen imposible una evaluación consistente de esas copias idénticas sin una resolución explícita. Esta limpieza se refiere a duplicados *exactos*; no demuestra ausencia de imágenes casi idénticas o de series fotográficas correlacionadas.

## 5. Preparación y partición de los datos

El contrato experimental establece una división **70/15/15 estratificada** con semilla `split_seed=2026`. La estratificación busca conservar aproximadamente la proporción de clases en `train`, `val` y `test`; por el redondeo y el tamaño de cada clase, no implica cantidades exactamente proporcionales.

En los cuadernos 04–06, la evaluación reporta **1.766 imágenes de entrenamiento** y **378 de validación**. El conjunto curado contiene 2.523 imágenes; si las particiones no tienen solapamientos ni exclusiones adicionales, las **379 restantes corresponderían a test**. Esta última cantidad es una inferencia aritmética, no una lectura de la carpeta de test en el equipo del usuario.

| Partición | Cantidad | Papel metodológico |
|:--|--:|:--|
| Train | 1.766 | Ajustar los parámetros de la CNN mediante gradientes |
| Val | 378 | Seleccionar checkpoint, comparar configuraciones y analizar errores |
| Test | 379 (inferido) | Evaluación final, una vez elegida la configuración |
| Total curado | 2.523 | Fotografías tras curación |

El `manifest.csv` y `dataset_stats.json` de `data/processed/` permiten auditar la partición y la transformación. Se recomienda verificar por identificador de imagen que no exista solapamiento entre conjuntos, así como comprobar que las 11 clases figuran en cada partición.

## 6. Transformación de imágenes y normalización

El pipeline descrito en los documentos de origen utiliza `ImageFolder`, conversión a RGB, `Resize((256,256))`, `ToTensor()` y `Normalize(mean_train, std_train)`. El tensor de una imagen tiene forma `(3, 256, 256)` y el lote tiene forma `(B, 3, 256, 256)`.

Para cada canal RGB, la media y la desviación estándar se calculan **únicamente con train**:

\[
\mu_c=\frac{1}{N_c}\sum_{i=1}^{N_c}x_{ic},\qquad
\sigma_c=\sqrt{\frac{1}{N_c}\sum_{i=1}^{N_c}(x_{ic}-\mu_c)^2},
\qquad z_{ic}=\frac{x_{ic}-\mu_c}{\sigma_c}.
\]

Esta restricción evita incorporar información estadística de validación o test en la preparación del modelo (*data leakage*). La normalización no garantiza igualar los contrastes de todas las fotografías; estandariza la escala de los canales respecto de train. El protocolo original no incluye aumentación de datos.

## 7. Experimento 02: variante con recorte inferior

El notebook 02 crea una versión recortada del dataset **sin sobrescribir `data/processed/`**. Conserva las particiones y etiquetas, modifica la región visible de la fotografía y **recalcula media y desviación estándar usando solamente el train recortado**. Esto es indispensable: la distribución de intensidades de una imagen recortada puede diferir de la original.

El recorte es una intervención sobre la información de entrada, no sobre los pesos de la CNN. Su hipótesis implícita es que cierta región inferior puede aportar ruido o elementos no relevantes; también existe el riesgo de eliminar contexto meteorológico útil. El notebook documenta la creación, la comparación visual y el entrenamiento, pero la versión adjunta no conserva resultados ejecutados de ese entrenamiento. Por tanto, no se le atribuye una mejora o pérdida cuantificada.

## 8. Controles de reproducibilidad y límites

1. Conservar `data/raw/` como fuente inmutable y registrar exclusiones.
2. Utilizar el mismo manifiesto de partición en los experimentos que comparan arquitecturas.
3. Calcular estadísticas exclusivamente sobre el train correspondiente a cada variante de entrada.
4. Registrar `split_seed`, `train_seed`, versiones de dependencias y hardware.
5. Distinguir `center_crop=None` de un recorte inferior materializado en otra carpeta.
6. No utilizar test para escoger épocas, hiperparámetros ni arquitecturas.
7. Conservar los CSV y JSON generados junto con el nombre inequívoco del experimento.




