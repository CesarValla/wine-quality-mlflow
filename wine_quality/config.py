from pathlib import Path

# Rutas del proyecto
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Archivos de datos
RED_WINE_FILE = RAW_DATA_DIR / "winequality-red.csv"
WHITE_WINE_FILE = RAW_DATA_DIR / "winequality-white.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "vinos_limpio.csv"

# Columnas que usa el modelo
TARGET = "quality"
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "es_tinto",
]

# Limpieza: límite para quitar el valor extremo de azúcar residual encontrado en el EDA
MAX_RESIDUAL_SUGAR = 60

# Valores por defecto del experimento
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42
DEFAULT_TRACKING_URI = "http://127.0.0.1:8080"
DEFAULT_EXPERIMENT_NAME = "wine-quality-elasticnet"
