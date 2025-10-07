# Clasificador-de-N-meros

# README — Análisis comparativo (MNIST)

> **Resumen:** Dos notebooks(ambos usan `tensorflow_datasets` y MNIST). Uno entrena una **CNN** (`NumerosMejorada.ipynb`) y el otro una **red densa** (`RedesNeuonales (Regular).ipynb`). Este README describe qué hace cada script/notebook, extrae los resultados de entrenamiento y compara ambos modelos estrictamente con lo que hay en los archivos.

---

## Contenido analizado

* `NumerosMejorada.ipynb` — implementación y entrenamiento de una **red convolucional (CNN)**.
* `RedesNeuonales (Regular).ipynb` — implementación y entrenamiento de una **red totalmente conectada (MLP / densa)**.

---

## Preparación de datos (común a ambos notebooks)

* Dataset: `mnist` cargado con `tfds.load('mnist', as_supervised=True, with_info=True)`.
* Preprocesado: casting a `tf.float32` y normalización `image /= 255.0`.
* Pipeline de entrenamiento: `cache()`, `repeat()`, `shuffle(num_train)`, `batch(batch_size)`, `prefetch`.
* Batch size: **32**.
* Épocas: **30**.
* Steps por época: **1875** (60,000 / 32 → ceil).
* Validation steps: **313** (10,000 / 32 → ceil).
* Loss: `SparseCategoricalCrossentropy`.
* Optimizer: `Adam` (lr por defecto).
* Métrica: `accuracy`.

---

## Arquitecturas

### CNN (`NumerosMejorada.ipynb`)

```text
Sequential([
  Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'),
  MaxPooling2D(2,2),
  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D(2,2),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(10, activation='softmax')
])
```

* Parámetros aproximados totales: **~225,034**.

### MLP / Densa (`RedesNeuonales (Regular).ipynb`)

Sequential([
  Flatten(input_shape=(28,28,1)),
  Dense(50, activation='relu'),
  Dense(50, activation='relu'),
  Dense(10, activation='softmax')
])

---

## Resultados extraídos

### CNN (NumerosMejorada.ipynb)

* Mejor `val_accuracy` (máximo observado): **0.9931** (99.31%) — **época 18**.
* `val_loss` en esa época: **0.0428**.
* `val_accuracy` al final del entrenamiento (época 30): **0.9892** (98.92%).
* `val_loss` final: **0.0658**.
* `accuracy` de entrenamiento final: **0.9996**.
  
* Observación: pico de validación alrededor de la época 18, luego ligera caída (ligero sobreajuste posterior al pico).

### MLP / Densa (RedesNeuonales (Regular).ipynb)

* Mejor `val_accuracy` (máximo observado): **0.9747** (97.47%) — **época 26**.
* `val_loss` en esa época: **0.1511**.
* `val_accuracy` al final del entrenamiento (época 30): **0.9729** (97.29%).
* `val_loss` final: **0.1778**.
* `accuracy` de entrenamiento final: **0.9963**.

* Observación: entrenamiento con alta accuracy, pero validación consistente y peor que la CNN.

---

## Comparación directa

* **Precisión máxima en validación:**

  * CNN: **99.31%** (mejor época).
  * Densa: **97.47%** (mejor época).
    
  * **Diferencia aproximada:** ~**1.8–2 puntos porcentuales** a favor de la CNN.

* **Pérdida en validación (val_loss):**

  * CNN (mejor época): **0.0428**.
  * Densa (mejor época): **0.1511**.
    
  * La CNN alcanza una `val_loss` marcadamente inferior.

* **Parámetros:**

  * CNN ≈ **225k** parámetros.
  * Densa ≈ **42k** parámetros.
  * La CNN usa ≈5× más parámetros y captura la estructura espacial de las imágenes.

* **Generalización / Overfitting:**

  * Ambos alcanzan accuracy de entrenamiento muy alto (>99%).
  * La CNN generaliza mejor (mayor `val_accuracy`, menor `val_loss`).
  * La CNN muestra leve sobreajuste tras su mejor época; la densa es más estable pero con tope inferior.

---

## Conclusión

* La **CNN** (`NumerosMejorada.ipynb`) logra **mejor desempeño** en MNIST que la **MLP** (`RedesNeuonales (Regular).ipynb`): mayor `val_accuracy` y menor `val_loss`.
* La diferencia en métricas valida empíricamente que —para MNIST y esas arquitecturas— la convolucional extrae y aprovecha la información espacial mejor que la pura densa, a costa de más parámetros.

