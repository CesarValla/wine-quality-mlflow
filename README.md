# wine-quality-mlflow

Experimentación con MLflow sobre el dataset Wine Quality (Cortez et al., 2009). El script entrena un modelo ElasticNet para predecir la calidad del vino y registra cada corrida en MLflow Tracking con sus parámetros, métricas, artefactos y el modelo entrenado.

Proyecto creado con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

## Estructura

```
├── data/raw/              <- Datos originales (no se modifican)
├── notebooks/             <- Notebook de exploración de datos (EDA)
├── reports/figures/       <- Gráficas generadas por el script
├── requirements.txt       <- Dependencias con versiones fijas
└── wine_quality/
    ├── config.py          <- Rutas, columnas y valores por defecto
    ├── dataset.py         <- Carga y limpieza de los datos
    ├── features.py        <- Separación de variables y división en entrenamiento y prueba
    ├── plots.py           <- Gráfica de calidad real contra predicha
    └── modeling/
        └── train.py       <- Entrenamiento, evaluación y registro en MLflow
```

## Requisitos

Python 3.14

## Instalación

```bash
git clone https://github.com/CesarValla/wine-quality-mlflow.git
cd wine-quality-mlflow
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux o Mac
pip install -r requirements.txt
```

## Ejecución

1. Levantar el servidor de MLflow en otra terminal y dejarla abierta:

```bash
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db
```

2. Correr el entrenamiento con los valores por defecto:

```bash
python -m wine_quality.modeling.train
```

3. O cambiar los hiperparámetros desde la línea de comandos:

```bash
python -m wine_quality.modeling.train --alpha 0.1 --l1-ratio 0.9 --max-iter 500 --seed 42
```

Para ver todos los argumentos: `python -m wine_quality.modeling.train --help`

Los resultados se ven en http://127.0.0.1:8080, en el experimento `wine-quality-elasticnet`.

## Argumentos

| Argumento | Default | Descripción |
|---|---|---|
| `--alpha` | 0.01 | Fuerza de la regularización. Cerca de 0 se comporta como una regresión lineal. |
| `--l1-ratio` | 0.5 | Mezcla de regularización: 0 es solo L2 (Ridge) y 1 es solo L1 (Lasso). |
| `--max-iter` | 1000 | Número máximo de iteraciones para ajustar el modelo. |
| `--seed` | 42 | Semilla para la división de datos y el modelo. |
| `--test-size` | 0.2 | Proporción de los datos que se usa para prueba. |
| `--experiment-name` | wine-quality-elasticnet | Nombre del experimento en MLflow. |
| `--tracking-uri` | http://127.0.0.1:8080 | Dirección del servidor de MLflow. |

## Qué registra cada corrida

1. **Parámetros:** alpha, l1_ratio, max_iter, seed, test_size y número de registros de entrenamiento y prueba.
2. **Métricas:** MAE, RMSE, R², exactitud (predicción redondeada), tolerancia de 1 punto, MAE de entrenamiento e iteraciones usadas.
3. **Artefactos:** `prediccion_vs_real.png` y `metricas.json`.
4. **Modelo:** el Pipeline completo (escalado + ElasticNet) con `mlflow.sklearn.log_model`.

## Reproducibilidad

La semilla (`--seed`) controla la división de los datos y el modelo, y las dependencias están fijadas en `requirements.txt`. Dos ejecuciones con los mismos argumentos producen exactamente las mismas métricas.
