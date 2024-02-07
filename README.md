# Estudio de la predictibilidad del tráfico en Internet para la detección de anomalías sutiles
_Repositorio basado en el Trabajo de Final de Máster con título "Estudio de la predictibilidad del tráfico en Internet para la detección de anomalías sutiles"_

El sistema completo consta de tres etapas claramente diferenciadas. La primera consiste en la extracción de los datos de entrenamiento; la segunda es el núcleo del algoritmo de predicción; la última consiste en el algoritmo con el que se utiliza el modelo predicho para evaluar anomalías sutiles sobre la serie temporal.

![General System Diagram](/readme_images/system-general.png)

# Índice
1. [Extracción de datos de entrenamiento](#extraccion-de-datos-de-entrenamiento)

    1. [Conversor de flujos de red a series temporales](#conversor-de-flujos-de-red-a-series-temporales)
    2. [Extractor de parámetros estadísticos y coeficientes polinómicos](#extractor-de-parámetros-estadísticos-y-coeficientes-polinómicos)

2. [Algoritmo de predicción](#algoritmo-de-entrenamiento)
3. [Algoritmo de evaluación](#algoritmo-de-evaluación)

## Extracción de datos de entrenamiento
El primer paso consiste en extraer la Base de Datos que se utilizará para el entrenamiento posterior. El directorio con el código para obtener estos datos es ```/Data_extraction```. Consta de los siguientes subdirectorios y ficheros:
- ```/NetflowsToTimeSeries```: conversor de flujos de red a series temporales
- ```TrendDynamics.m```: extractor de parámetros estadísticos y coeficientes polinómicos para el entrenamiento
- ```/Functions```: dependencias de ```TrendDynamics.m```

### Conversor de flujos de red a series temporales
```/NetflowsToTimeSeries/netflows2timeseries.sh```
Consiste en un Script Bash escrito en AWK que convierte los ficheros con información de flujos de red en series temporales de ancho de banda.

- **Inputs**: ficheros con la información de flujos de red con extensión ```.csv.uniqblacklistremoved```. Es posible descargarlos en [UGR'16](https://nesg.ugr.es/nesg-ugr16/index.php) (formato ```.tar.gz```) y descomprimirlos
- **Outputs**: ficheros de salida del procesamiento con la información de ancho de banda (tanto en unidades de paquetes por segundo como bits por segundo) con extensión ```.csv```

Para ello, es necesario guardar dentro de ```/Input_files``` los ficheros de flujos con extensión ```.csv.uniqblacklistremoved``` que se desean procesar y ejecutar el siguiente comando:

```shell
./netflows2timeseries.sh
```

La salida se guardará automáticamente en ```/Ouput_files```.

### Extractor de parámetros estadísticos y coeficientes polinómicos
```/NetflowsToTimeSeries/TrendDynamics.m```
Código escrito en Matlab para la obtención de la Base de Datos de entrenamiento con las **dinámicas de la serie temporal** (i.e. los parámetros estadísticos y los coeficientes polinómicos; sin embargo, en este trabajo solo se explora la predicción de coeficientes polinómicos). Es necesario ejecutarlo desde el entorno de Matlab.
Las variables de entrada del fichero TrendDynamics.m son:
- ```filenames```: ficheros de entrada de los que se desea obtener las dinámicas de la serie temporal
- ```JSONoutput_filename```: nombre del fichero de salida con los datos de entrenamiento, que será un fichero con extensión ```.JSON```
- ```computeParams``` (boolean: 0 o 1) para decidir si leer los ficheros TP y AP de un .txt ya existente (```computeParams = 0```) o bien computarlos y escribirlos (```computeParams = 1```)
- ```Tventana [min]```: tamaño en minutos de la ventana deslizante T, medido en minutos
- ```n```: Grado para la regresión polinómica
- ```Granularidad_deteccion [s]```: es el alcance del sistema, medido en segundos, que se debe conocer para el dominio en el que realizar la regresión polinómica. En otras palabras, cuánto tiempo futuro se desea predecir
- ```bitsPaquetes``` (2 o 3): indica si se desea extraer las dinámicas de la serie temporal en [bits/s] (2) o [packets/s] (3)

Dichas variables han de ser editadas directamente en el Script ```TrendDynamics.m```.
La salida del Script será un fichero ```.JSON``` con los datos de entrenamiento para la red neuronal LSTM.

## Algoritmo de predicción
```/Network_forecasting/main.py```

Código Python para la programación de la red LSTM y para la predicción de la tendencia (i.e. de cada coeficiente del polinomio en la ventana de test).
Para ejecutarlo:

```shell
python main.py
```

## Algoritmo de evaluación
```/Anomaly_evaluation/resultadosFinalesAlgoritmo.m```

