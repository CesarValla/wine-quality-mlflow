from matplotlib.figure import Figure
import numpy as np


def graficar_predicciones(y_real, y_pred, ruta_salida):
    """Guarda un boxplot de la calidad predicha según la calidad real."""
    y_real = np.asarray(y_real)
    calidades = np.unique(y_real)
    grupos = [y_pred[y_real == calidad] for calidad in calidades]

    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    ax.boxplot(grupos, tick_labels=calidades)
    posiciones = range(1, len(calidades) + 1)
    ax.plot(posiciones, calidades, "o--", color="#eb6834", label="predicción perfecta")
    ax.set_title("Calidad predicha según la calidad real (prueba)")
    ax.set_xlabel("quality real")
    ax.set_ylabel("quality predicha")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, bbox_inches="tight")
    return ruta_salida
