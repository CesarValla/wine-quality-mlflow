import pandas as pd

from wine_quality.config import (
    MAX_RESIDUAL_SUGAR,
    PROCESSED_DATA_FILE,
    RED_WINE_FILE,
    WHITE_WINE_FILE,
)


def cargar_datos():
    """Lee los archivos de vinos tintos y blancos y los junta en una sola tabla."""
    tinto = pd.read_csv(RED_WINE_FILE, sep=";")
    blanco = pd.read_csv(WHITE_WINE_FILE, sep=";")
    tinto["tipo_vino"] = "tinto"
    blanco["tipo_vino"] = "blanco"
    return pd.concat([tinto, blanco], ignore_index=True)


def limpiar_datos(vinos):
    """Aplica las decisiones de limpieza documentadas en el notebook de exploración."""
    limpio = vinos.drop_duplicates()
    limpio = limpio[limpio["residual sugar"] < MAX_RESIDUAL_SUGAR].copy()
    limpio["es_tinto"] = (limpio["tipo_vino"] == "tinto").astype(int)
    return limpio.reset_index(drop=True)


def main():
    vinos = limpiar_datos(cargar_datos())
    PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    vinos.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Datos limpios guardados en {PROCESSED_DATA_FILE} ({len(vinos)} registros)")


if __name__ == "__main__":
    main()
