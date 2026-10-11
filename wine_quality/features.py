from sklearn.model_selection import train_test_split

from wine_quality.config import FEATURES, TARGET


def separar_variables(vinos):
    """Separa las variables de entrada (X) de la variable objetivo (y)."""
    return vinos[FEATURES], vinos[TARGET]


def dividir_datos(X, y, test_size, seed):
    """Divide en entrenamiento y prueba manteniendo la proporción de cada calidad."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
