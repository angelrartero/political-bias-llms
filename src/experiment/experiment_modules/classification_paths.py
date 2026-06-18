# Construye rutas de salida para guardar clasificaciones fuera de responses.
from __future__ import annotations

from pathlib import Path

from experiment_modules.paths import CLASSIFICATIONS_DIR, RESPONSES_DIR


def infer_model_name_from_response_csv(input_csv: str | Path) -> str:
    """Deduce el alias del modelo a partir de la ubicacion del CSV."""
    input_path = Path(input_csv)

    try:
        # Si el CSV esta dentro de data/responses/<modelo>/, usa esa carpeta
        # como nombre para ordenar las clasificaciones por modelo.
        relative_input = input_path.resolve().relative_to(RESPONSES_DIR.resolve())
        if len(relative_input.parts) > 1:
            return relative_input.parts[0]
    except ValueError:
        pass

    return input_path.parent.name or "unknown_model"


def build_classified_output_csv(input_csv: str | Path, prefix: str = "classified_") -> Path:
    """Construye la ruta del CSV clasificado dentro de data/classifications."""
    input_path = Path(input_csv)
    model_name = infer_model_name_from_response_csv(input_path)
    return CLASSIFICATIONS_DIR / model_name / f"{prefix}{input_path.name}"


def build_summary_json(output_csv: str | Path) -> Path:
    """Usa la misma ruta del CSV de salida cambiando la extension a JSON."""
    return Path(output_csv).with_suffix(".summary.json")
