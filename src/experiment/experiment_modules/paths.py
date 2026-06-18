# Define las rutas base del proyecto usadas por los experimentos.
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Carpetas compartidas por los scripts de generacion, clasificacion y resumen.
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
RESPONSES_DIR = DATA_DIR / "responses"
CLASSIFICATIONS_DIR = DATA_DIR / "classifications"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
