# ia-neuraltraining-mnist-py

Adaptación del ejercicio MNIST del profesor para ejecutarlo localmente con Python 3.14, sin Marimo. El código está en inglés y no contiene comentarios.

## Ejecutar

### Página visual

```bash
uv sync --locked
uv run python dashboard.py
```

Abre **http://127.0.0.1:8000** en tu navegador. Mantén el proceso abierto mientras usas la página; para cerrarlo, presiona `Ctrl+C` en la terminal.

La navegación superior contiene tres módulos:

- **Red neuronal**: entrenamiento y experimentos que ya utilizábamos.
- **KNN**: clasificador manual de vecinos, con MNIST y el ejemplo Iris del profesor.
- **Comparar métodos**: evaluación de una red guardada frente a KNN sobre exactamente las mismas imágenes de MNIST.

### KNN

Abre **http://127.0.0.1:8000/knn**. El algoritmo adapta la distancia euclidiana y la votación uniforme de `knn_iris.py` del profesor. No utiliza Marimo. El archivo del profesor permanece intacto. Scikit-learn se utiliza para cargar Iris y crear su partición estratificada; la búsqueda de vecinos y la votación se implementan con NumPy.

Puedes:

- Elegir **MNIST** o **Iris**, modificar K o comparar hasta 20 valores de K entre 1 y 50.
- Elegir cuántas referencias de MNIST conservar y cuántas imágenes reservadas evaluar. Para compararlo con las redes existentes, utiliza las 50.000 referencias completas.
- Repetir una selección con la misma semilla. Las selecciones parciales de MNIST conservan aproximadamente las proporciones de los dígitos; las imágenes de validación nunca entran en la referencia.
- En Iris, elegir la fracción de validación y activar la estandarización. Media y escala se ajustan exclusivamente sobre entrenamiento. El valor predeterminado usa una fracción del 20 %; con semilla 42 reproduce la partición del profesor.
- Ver registros, progreso, una gráfica de aciertos frente a K, matriz de confusión y precisión, recall y F1 por clase.
- Probar un dígito de MNIST por índice o ingresar las cuatro medidas de una flor. Se muestran los votos y hasta 12 vecinos, aunque la decisión utiliza todos los K seleccionados.
- Recuperar resultados, descargarlos como JSON o borrar una ejecución o todas las del módulo, con confirmación en pantalla.

KNN no tiene alpha, pesos ni epochs: `fit` conserva referencias y la predicción busca los ejemplos cercanos. El código procesa las consultas en lotes de 64 para evitar crear una matriz de distancias de todas las imágenes a la vez. Un barrido reutiliza los mismos vecinos para todos los K. Su tiempo corresponde al barrido completo, no a un entrenamiento iterativo ni a una medición independiente por cada K.

Los vecinos se ordenan por distancia y, cuando empatan, por su posición de referencia. Si varias clases empatan en votos, gana la que aparece primero entre los vecinos de las clases empatadas, como con `Counter(...).most_common(1)` en el ejemplo del profesor. Las proporciones de votos no son probabilidades calibradas.

### Comparar KNN con la red neuronal

Abre **http://127.0.0.1:8000/compare**:

1. Prepara una ejecución KNN de **MNIST con 50.000 referencias**.
2. Selecciona una red neuronal guardada y esa ejecución KNN.
3. Elige K, la cantidad de imágenes de validación y la semilla de selección.
4. Ejecuta la comparación. La red no se vuelve a entrenar.

El servidor exige la misma partición y todas las referencias de entrenamiento para ambos métodos. Rechaza Iris y los KNN preparados con menos referencias. Ambos reciben los mismos píxeles entre 0 y 1 y exactamente los mismos índices de validación. La comparación vuelve a predecir con ambos; no enfrenta una métrica histórica sobre 10.000 imágenes con otra calculada sobre 1.000.

Los resultados incluyen aciertos, errores, F1 promedio, matrices de confusión, coincidencias y desacuerdos, ejemplos de imágenes y tiempos de predicción. Se usa el mismo tamaño de lote, un calentamiento inicial y un orden alternado entre métodos. El promedio por imagen es rendimiento por lotes, no latencia de una petición individual. Los tiempos dependen del equipo y su carga.

La memoria suma los arrays numéricos que utiliza cada modelo: pesos y biases de la red frente a referencias, etiquetas y arrays auxiliares de KNN. No representa toda la RAM del proceso ni el tamaño del archivo guardado. El tiempo de entrenamiento de la red se recupera de su ejecución original; en KNN se muestra el tiempo necesario para preparar las referencias. No se presenta una pérdida softmax como equivalente a los votos de KNN.

Los tres módulos conservan sus resultados separados. KNN y las comparaciones se guardan en `outputs/web/lab/<id>/run.json`, con configuración, índices, predicciones y métricas. KNN se reconstruye al reiniciar usando MNIST en caché o Iris y los índices guardados, sin duplicar las 50.000 imágenes por experimento. Se conserva en memoria un solo modelo KNN reconstruido. Borrar estos resultados no borra datasets ni redes neuronales; una comparación ya guardada conserva sus métricas aunque se elimine su ejecución KNN de origen.

Se permite una evaluación KNN o comparación a la vez. El entrenamiento neuronal tampoco puede ejecutarse simultáneamente, para evitar competir por CPU durante las mediciones. Puedes detener una evaluación entre lotes; una ejecución detenida no se presenta como resultado completo ni se guarda parcialmente. Los resultados anteriores se conservan.

Todos estos resultados siguen siendo **validación**, incluida la selección de K. Para una evaluación final hay que reservar un conjunto de test independiente.

### Controles de la red neuronal

Desde la página puedes:

- Cambiar alpha (learning rate), iteraciones, semilla y el tamaño de la primera capa oculta (de 1 a 256 neuronas).
- Activar una segunda capa oculta y elegir su tamaño. La arquitectura y la cantidad de parámetros se muestran automáticamente.
- Preparar los experimentos sugeridos: red base, alpha 0.1, alpha 1, 32 neuronas, 64 neuronas o dos capas de 32 y 16 neuronas.
- Ver logs y curvas de precisión y pérdida de entrenamiento y validación. Por defecto se registra cada iteración y se mide validación cada 50; ambos intervalos se pueden cambiar. La validación también se mide al inicio, al terminar y al detener un entrenamiento.
- Detener el entrenamiento y conservar los pesos obtenidos hasta ese momento.
- Probar una imagen de MNIST por índice, o buscar otra imagen de un dígito del 0 al 9.
- Subir una imagen PNG, JPG o WebP con un solo dígito sobre un fondo uniforme. Se convierte a escala de grises, se recorta y se centra en 28 × 28 píxeles. La página muestra exactamente la imagen que recibe la red.
- Ver la predicción, su probabilidad y la distribución completa entre los diez dígitos. Puedes elegir qué modelo guardado usar para probar la misma imagen.
- Comparar hasta cuatro experimentos con curvas superpuestas y una tabla de alpha, arquitectura, iteraciones realizadas, semilla, parámetros, precisión, pérdida de validación y tiempo.
- Identificar cada prueba con un nombre automático que incluye alpha, capas ocultas, iteraciones, semilla y un identificador. Los nombres antiguos se corrigen al arrancar según la configuración guardada.
- Recuperar la configuración de cualquier experimento con Usar valores, modificar una variable y volver a entrenar.
- Borrar un experimento con Borrar, o vaciar el historial con Borrar todos los experimentos. Ambos muestran una confirmación antes de eliminar pesos, métricas y logs. Durante la carga o un entrenamiento no se permite borrar.

La página utiliza el último modelo guardado al arrancar. Durante un entrenamiento, las predicciones siguen usando el último modelo finalizado; al terminar, se actualizan automáticamente. Los cambios de alpha se aplican cuando inicias otro entrenamiento.

Cada experimento se conserva en `outputs/web/experiments/<id>/`, con `model.npz` y `run.json` (configuración, métricas, estado, duración e historia de las curvas). Las ejecuciones nuevas no reemplazan los experimentos anteriores.

Los archivos `model.npz`, `history.csv` y `run.json` de `outputs/web/` siguen señalando el último entrenamiento. Al reiniciar se recuperan las arquitecturas, los logs y la lista de experimentos. El modelo existente anterior a esta actualización se importa una sola vez como Modelo anterior: su validación histórica solo está disponible en el punto final; no se inventan puntos intermedios.

Todo se guarda en archivos locales, sin base de datos ni S3. `model.npz` contiene los tensores de pesos y biases; `run.json` contiene configuración, métricas e historia de las curvas; `history.csv` es una copia tabular del historial del modelo actual. Los datos originales de MNIST se descargan y reutilizan en `data/`. Las imágenes propias se procesan en memoria, sin guardarse en disco.

Si borras el modelo actual, la página recupera el experimento más reciente que todavía exista. Al borrar el último o todos, se limpian también las copias del modelo actual y se deja el dashboard sin modelo. `outputs/web/reset.json` registra ese vaciado para evitar volver a importar el modelo de terminal al reiniciar. Se conservan MNIST, las capturas de pantalla y los resultados separados del programa de terminal en `outputs/`. Después puedes entrenar de nuevo desde la página.

El servidor escucha únicamente en tu computador. Para usar otro puerto:

```bash
uv run python dashboard.py --port 8001
```

Las probabilidades representan la salida del modelo, no una garantía de acierto. Las imágenes propias deben parecerse a dígitos manuscritos de MNIST; una foto completa con fondo complejo puede dar una predicción incorrecta.

### Programa de terminal

Desde la carpeta del proyecto, con `uv` instalado:

```bash
uv sync --locked
uv run python mnist.py
```

`uv` utiliza Python 3.14 y las dependencias del entorno `.venv`. No es necesario cambiar el Python global del equipo. La primera ejecución descarga MNIST en `data/`; las siguientes reutilizan esos archivos.

El programa entrena la red, imprime el progreso y la precisión de evaluación, guarda los resultados y abre dos ventanas de Matplotlib. Cierra las ventanas para terminar la ejecución.

Para guardar las gráficas sin abrir ventanas:

```bash
uv run python mnist.py --no-show
```

Para comprobar la instalación con una ejecución corta:

```bash
uv run python mnist.py --iterations 5 --log-every 1 --no-show --output-dir outputs/smoke
```

Para elegir una imagen distinta en la demostración:

```bash
uv run python mnist.py --image-index 50001
```

Las opciones disponibles se muestran con `uv run python mnist.py --help`. El archivo original del repositorio, `mnsit.py`, también funciona y ejecuta el mismo programa.

## Qué entrena

- La página admite `784 → H1 → 10` o `784 → H1 → H2 → 10`. El programa de terminal conserva la red original `784 → 10 → 10`.
- Las capas ocultas usan sigmoid y las 10 salidas usan softmax. Se conserva la inicialización normal a escala 0.01 del profesor para mantener controlada esa variable al comparar arquitecturas.
- Forward, backpropagation y actualización de pesos implementados manualmente con NumPy, como en el ejercicio del profesor.
- 500 iteraciones, learning rate de 0.5 y semilla de 0 por defecto.
- Cada actualización utiliza las 50.000 imágenes de entrenamiento juntas: full-batch gradient descent, ejecutado en CPU.
- Pérdida de entropía cruzada, correspondiente al gradiente de softmax del ejercicio.
- PyTorch y Torchvision cargan el dataset; el entrenamiento no usa autograd ni la GPU.

Se conserva la partición del profesor: las primeras 50.000 imágenes del conjunto de entrenamiento de MNIST se usan para entrenar y las otras 10.000 para evaluar. Esta evaluación es un conjunto reservado dentro del training original, no el conjunto oficial de test de MNIST. El dashboard utiliza esa partición reservada como validación para comparar hiperparámetros. Las métricas son de validación, no del test oficial.

Se corrigió softmax para restar el máximo de cada imagen individualmente y se utiliza una sigmoid estable. Los cálculos usan `float32` para reducir memoria. El archivo del profesor en Downloads no se modifica.

## Resultados

Cada ejecución guarda en `outputs/`:

| Archivo | Contenido |
| --- | --- |
| `training.png` | Precisión y pérdida durante el entrenamiento |
| `prediction.png` | Imagen elegida y probabilidades para los dígitos del 0 al 9 |
| `model.npz` | Pesos y biases aprendidos: `W1`, `b1`, `W2`, `b2` |
| `history.csv` | Iteración, precisión y pérdida |

Una nueva ejecución reemplaza estos resultados. Usa `--output-dir outputs/run-2` para conservar una ejecución anterior. El modelo se guarda en formato NumPy `.npz`, porque esta red se entrena con NumPy.

## Ejecutar desde el IDE

Selecciona `.venv/bin/python` como intérprete del proyecto y ejecuta `mnist.py`. Los directorios predeterminados de datos y resultados se resuelven desde el archivo, por lo que no dependen de la carpeta desde la que lo ejecutes.

## Verificación

```bash
uv run python -m unittest discover -s tests -v
```

Las pruebas verifican gradientes mediante diferencias finitas para una y dos capas ocultas, equivalencia con la red original, validación sin modificar pesos, recuperación de arquitecturas y experimentos, predicciones con modelos anteriores, logs, selección de dígitos, imágenes propias, validación de valores, detención de entrenamientos y borrado individual o completo con recuperación correcta al reiniciar. También contrastan KNN con la implementación directa del profesor y con Scikit-learn en Iris, prueban empates, estandarización, barridos de K, recuperación de referencias, cancelación y comparación sobre los mismos índices sin modificar los pesos guardados.
