import argparse
import json

import mlflow
from mlflow.models import infer_signature
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wine_quality.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRACKING_URI,
    FIGURES_DIR,
    REPORTS_DIR,
)
from wine_quality.dataset import cargar_datos, limpiar_datos
from wine_quality.features import dividir_datos, separar_variables
from wine_quality.plots import graficar_predicciones


def leer_argumentos():
    """Define los argumentos de línea de comandos con sus valores por defecto."""
    parser = argparse.ArgumentParser(
        description="Entrena un modelo ElasticNet para predecir la calidad del vino "
        "y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Fuerza de la regularización. Cerca de 0 se comporta como una regresión lineal.",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.5,
        help="Mezcla de regularización: 0 es solo L2 (Ridge) y 1 es solo L1 (Lasso).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Número máximo de iteraciones para ajustar el modelo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla para la división de datos y el modelo.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proporción de los datos que se usa para prueba.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Nombre del experimento en MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="Dirección del servidor de MLflow.",
    )
    return parser.parse_args()


def construir_modelo(alpha, l1_ratio, max_iter, seed):
    """Escalado + ElasticNet dentro de un Pipeline."""
    modelo = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=seed)
    return Pipeline([("escalado", StandardScaler()), ("modelo", modelo)])


def calcular_metricas(y_real, y_pred):
    """Métricas de regresión y de acierto sobre la calidad redondeada."""
    return {
        "mae": mean_absolute_error(y_real, y_pred),
        "rmse": root_mean_squared_error(y_real, y_pred),
        "r2": r2_score(y_real, y_pred),
        "exactitud": float(np.mean(np.round(y_pred) == y_real)),
        "tolerancia_1": float(np.mean(np.abs(y_pred - y_real) <= 1)),
    }


def guardar_metricas(metricas, ruta_salida):
    """Guarda las métricas en un archivo JSON para subirlo como artefacto."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    return ruta_salida


def main():
    args = leer_argumentos()
    np.random.seed(args.seed)

    vinos = limpiar_datos(cargar_datos())
    X, y = separar_variables(vinos)
    X_train, X_test, y_train, y_test = dividir_datos(X, y, args.test_size, args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    nombre_run = f"elasticnet_alpha{args.alpha}_l1{args.l1_ratio}_iter{args.max_iter}"
    with mlflow.start_run(run_name=nombre_run):
        modelo = construir_modelo(args.alpha, args.l1_ratio, args.max_iter, args.seed)
        modelo.fit(X_train, y_train)
        pred_prueba = modelo.predict(X_test)

        metricas = calcular_metricas(y_test, pred_prueba)
        metricas["mae_entrenamiento"] = mean_absolute_error(y_train, modelo.predict(X_train))
        metricas["iteraciones_usadas"] = int(modelo.named_steps["modelo"].n_iter_)

        mlflow.log_params(
            {
                "alpha": args.alpha,
                "l1_ratio": args.l1_ratio,
                "max_iter": args.max_iter,
                "seed": args.seed,
                "test_size": args.test_size,
                "registros_entrenamiento": len(X_train),
                "registros_prueba": len(X_test),
            }
        )
        mlflow.log_metrics(metricas)

        ruta_grafica = graficar_predicciones(
            y_test, pred_prueba, FIGURES_DIR / "prediccion_vs_real.png"
        )
        ruta_metricas = guardar_metricas(metricas, REPORTS_DIR / "metricas.json")
        mlflow.log_artifact(str(ruta_grafica))
        mlflow.log_artifact(str(ruta_metricas))

        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo",
            signature=infer_signature(X_train, modelo.predict(X_train)),
            input_example=X_train.head(5),
        )

    print(f"Corrida '{nombre_run}' registrada en el experimento '{args.experiment_name}'")
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor:.4f}")


if __name__ == "__main__":
    main()
