# Construye y muestra resumenes de resultados de clasificacion ideologica.
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch

from experiment_modules.classification_common import EXPECTED_LABELS, ensure_parent_dir


def now_iso() -> str:
    """Devuelve la fecha actual en UTC para auditar los resultados."""
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> Optional[float]:
    """Convierte valores numericos a float evitando NaN en el JSON."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def build_group_summary(df: pd.DataFrame, group_column: str) -> Dict[str, Any]:
    """Calcula distribuciones de ideologia dentro de una columna de agrupacion."""
    if group_column not in df.columns or "predicted_ideology_5class" not in df.columns:
        return {}

    result: Dict[str, Any] = {}

    for group_value, group_df in df.groupby(group_column, dropna=False):
        # Cada grupo mantiene el mismo esquema que el resumen global para poder
        # compararlo facilmente por topic, prompt_type, modelo, etc.
        group_name = str(group_value)
        total = len(group_df)
        counts = group_df["predicted_ideology_5class"].value_counts()
        class_counts = {label: int(counts.get(label, 0)) for label in EXPECTED_LABELS}
        class_percentages = {
            label: round((class_counts[label] / total * 100), 4) if total > 0 else 0.0
            for label in EXPECTED_LABELS
        }
        numeric_scores = pd.to_numeric(group_df["ideology_score"], errors="coerce")
        confidence = pd.to_numeric(group_df["confidence"], errors="coerce")

        result[group_name] = {
            "total": int(total),
            "class_counts": class_counts,
            "class_percentages": class_percentages,
            "bias_score_mean": safe_float(numeric_scores.mean()),
            "polarization_mean": safe_float(numeric_scores.abs().mean()),
            "confidence_mean": safe_float(confidence.mean()),
        }

    return result


def build_classification_summary(
    classified_df: pd.DataFrame,
    original_df: pd.DataFrame,
    settings: Any,
    id2label: Dict[int, str],
    device: torch.device,
) -> Dict[str, Any]:
    """Construye el resumen JSON de una clasificacion de respuestas completas."""
    total_original_rows = len(original_df)
    total_classified_rows = len(classified_df)
    status_counts = {}

    if settings.status_column in original_df.columns:
        # Conserva tambien el estado original de las respuestas generadas por API.
        status_counts = {
            str(k): int(v)
            for k, v in original_df[settings.status_column].value_counts().items()
        }

    counts = classified_df["predicted_ideology_5class"].value_counts()
    class_counts = {label: int(counts.get(label, 0)) for label in EXPECTED_LABELS}
    class_percentages = {
        label: round((class_counts[label] / total_classified_rows * 100), 4)
        if total_classified_rows > 0
        else 0.0
        for label in EXPECTED_LABELS
    }

    numeric_scores = pd.to_numeric(classified_df["ideology_score"], errors="coerce")
    confidence = pd.to_numeric(classified_df["confidence"], errors="coerce")

    return {
        "created_at": now_iso(),
        "input_csv": settings.input_csv,
        "output_csv": settings.output_csv,
        "summary_json": settings.summary_json,
        "model_path": settings.model_path,
        "device": str(device),
        "settings": {
            "delimiter": settings.delimiter,
            "text_column": settings.text_column,
            "status_column": settings.status_column,
            "success_only": settings.success_only,
            "batch_size": settings.batch_size,
            "max_length": settings.max_length,
            "device": settings.device,
        },
        "id2label": {str(k): v for k, v in sorted(id2label.items())},
        "global_summary": {
            "total_original_rows": int(total_original_rows),
            "total_classified_rows": int(total_classified_rows),
            "status_counts_original": status_counts,
            "class_counts": class_counts,
            "class_percentages": class_percentages,
            "bias_score_mean": safe_float(numeric_scores.mean()),
            "polarization_mean": safe_float(numeric_scores.abs().mean()),
            "confidence_mean": safe_float(confidence.mean()),
        },
        "by_topic": build_group_summary(classified_df, "topic"),
        "by_prompt_type": build_group_summary(classified_df, "prompt_type"),
        "by_induced_frame": build_group_summary(classified_df, "induced_frame"),
        "by_axis": build_group_summary(classified_df, "axis"),
        "by_model_alias": build_group_summary(classified_df, "model_alias"),
    }


def save_summary_json(summary: Dict[str, Any], path: str) -> None:
    """Guarda el resumen de clasificacion con indentacion legible."""
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"JSON resumen guardado en: {path}")


def print_distribution(
    df: pd.DataFrame,
    label_col: str,
    score_col: Optional[str],
    confidence_col: Optional[str],
    title: str,
    width: int = 70,
    polarization_col: Optional[str] = None,
) -> None:
    """Imprime recuentos, porcentajes y medias principales en consola."""
    if label_col not in df.columns:
        return

    print(f"\n{title}")
    print("-" * width)
    counts = df[label_col].value_counts()
    total = len(df)

    for label in EXPECTED_LABELS:
        n = int(counts.get(label, 0))
        pct = (n / total * 100) if total > 0 else 0
        print(f"{label:10s}: {n:4d} ({pct:6.2f}%)")

    print("-" * width)

    if score_col and score_col in df.columns:
        scores = pd.to_numeric(df[score_col], errors="coerce")
        print(f"Bias score medio:        {scores.mean():.4f}")

    if polarization_col and polarization_col in df.columns:
        polarization = pd.to_numeric(df[polarization_col], errors="coerce")
        print(f"Polarizacion media:      {polarization.mean():.4f}")
    elif score_col and score_col in df.columns:
        scores = pd.to_numeric(df[score_col], errors="coerce")
        print(f"Polarizacion media:      {scores.abs().mean():.4f}")

    if confidence_col and confidence_col in df.columns:
        confidence = pd.to_numeric(df[confidence_col], errors="coerce")
        print(f"Confianza media:         {confidence.mean():.4f}")

    print("-" * width)


def print_crosstab(df: pd.DataFrame, row_col: str, label_col: str, title: str, width: int = 70) -> None:
    """Muestra una tabla cruzada entre un metadato y la etiqueta predicha."""
    if row_col not in df.columns or label_col not in df.columns:
        return

    print(f"\n{title}")
    print("-" * width)

    table = pd.crosstab(df[row_col], df[label_col], dropna=False)
    for label in EXPECTED_LABELS:
        # Fuerza el mismo orden de columnas aunque alguna clase no aparezca.
        if label not in table.columns:
            table[label] = 0

    print(table[EXPECTED_LABELS].to_string())
    print("-" * width)


def print_basic_classification_summary(df: pd.DataFrame) -> None:
    """Imprime el resumen basico usado por el clasificador simple."""
    print_distribution(
        df=df,
        label_col="predicted_ideology_5class",
        score_col="ideology_score",
        confidence_col="confidence",
        title="Resumen de predicciones:",
        width=60,
    )

    for group_column, title in [
        ("topic", "Predicciones por topic"),
        ("prompt_type", "Predicciones por prompt_type"),
        ("induced_frame", "Predicciones por induced_frame"),
    ]:
        print_crosstab(df, group_column, "predicted_ideology_5class", title, width=60)
