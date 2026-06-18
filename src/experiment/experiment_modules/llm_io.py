# Gestiona lectura de prompts, progreso previo y escritura de resultados CSV.
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from experiment_modules.classification_common import ensure_parent_dir
from experiment_modules.llm_config import REQUIRED_PROMPT_COLUMNS


def now_iso() -> str:
    """Devuelve una marca temporal UTC para cada respuesta generada."""
    return datetime.now(timezone.utc).isoformat()


def read_prompts(path: str, delimiter: str = ";") -> pd.DataFrame:
    """Lee y valida el CSV de prompts que alimenta el experimento."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No existe el CSV de prompts: {path}")

    df = pd.read_csv(path, sep=delimiter, dtype=str).fillna("")
    missing = REQUIRED_PROMPT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "El CSV de prompts no tiene las columnas obligatorias: "
            + ", ".join(sorted(missing))
        )

    df["prompt_id"] = df["prompt_id"].astype(str).str.strip()
    df["prompt"] = df["prompt"].astype(str).str.strip()

    # prompt_id debe ser unico porque se usa para construir response_id y reanudar.
    if df["prompt_id"].duplicated().any():
        duplicates = df.loc[df["prompt_id"].duplicated(), "prompt_id"].tolist()
        raise ValueError(f"Hay prompt_id duplicados. Ejemplos: {duplicates[:10]}")

    empty_prompts = df[df["prompt"].str.len() == 0]
    if not empty_prompts.empty:
        raise ValueError(f"Hay prompts vacios: {empty_prompts['prompt_id'].tolist()[:10]}")

    return df


def load_completed_keys(output_csv: str, delimiter: str = ";") -> set[tuple[str, str, int]]:
    """Recupera llamadas success para poder evitar duplicados al reanudar."""
    path = Path(output_csv)
    if not path.exists():
        return set()

    try:
        df = pd.read_csv(path, sep=delimiter, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return set()

    required = {"prompt_id", "model_alias", "run", "status"}
    if not required.issubset(df.columns):
        return set()

    completed = df[df["status"] == "success"].copy()
    keys: set[tuple[str, str, int]] = set()

    for _, row in completed.iterrows():
        # La clave identifica prompt + modelo + repeticion.
        try:
            keys.add((str(row["prompt_id"]), str(row["model_alias"]), int(row["run"])))
        except ValueError:
            continue

    return keys


def load_existing_rows(output_csv: str, delimiter: str = ";") -> pd.DataFrame:
    """Carga filas existentes de respuestas si el CSV existe."""
    path = Path(output_csv)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=delimiter, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_retryable_keys(output_csv: str, delimiter: str = ";") -> set[tuple[str, str, int]]:
    """Recupera llamadas error/truncated que se deben reintentar al reanudar."""
    df = load_existing_rows(output_csv, delimiter)
    required = {"prompt_id", "model_alias", "run", "status"}
    if df.empty or not required.issubset(df.columns):
        return set()

    retryable = df[df["status"].isin(["error", "truncated"])].copy()
    keys: set[tuple[str, str, int]] = set()
    for _, row in retryable.iterrows():
        try:
            keys.add((str(row["prompt_id"]), str(row["model_alias"]), int(row["run"])))
        except ValueError:
            continue
    return keys


def drop_retryable_rows(output_csv: str, delimiter: str = ";") -> None:
    """Elimina filas error/truncated antes de reintentarlas para evitar duplicados."""
    df = load_existing_rows(output_csv, delimiter)
    if df.empty or "status" not in df.columns:
        return
    kept = df[~df["status"].isin(["error", "truncated"])].copy()
    if len(kept) == len(df):
        return
    ensure_parent_dir(output_csv)
    kept.to_csv(output_csv, sep=delimiter, index=False, encoding="utf-8")


def append_result_row(output_csv: str, row: Dict[str, Any], delimiter: str = ";") -> None:
    """Anade una respuesta al CSV de salida sin cargarlo entero en memoria."""
    ensure_parent_dir(output_csv)
    path = Path(output_csv)
    fieldnames = [
        # Mantiene un esquema estable para poder concatenar y clasificar salidas.
        "response_id",
        "created_at",
        "prompt_id",
        "topic",
        "prompt_type",
        "induced_frame",
        "axis",
        "nationalism_sensitive",
        "model_alias",
        "provider",
        "model",
        "run",
        "temperature",
        "max_tokens",
        "system_prompt",
        "prompt",
        "response",
        "status",
        "error_type",
        "error_message",
        "finish_reason",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "raw_usage_json",
    ]

    file_exists = path.exists() and path.stat().st_size > 0
    if file_exists:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            existing_header = next(reader, [])
        if existing_header and existing_header != fieldnames:
            df = pd.read_csv(path, sep=delimiter, dtype=str).fillna("")
            for field in fieldnames:
                if field not in df.columns:
                    df[field] = ""
            df = df[fieldnames]
            df.to_csv(path, sep=delimiter, index=False, encoding="utf-8")

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=delimiter,
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def normalize_usage(usage: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Normaliza contadores de tokens entre proveedores con nombres distintos."""
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")

    def to_int_or_none(value: Any) -> Optional[int]:
        """Convierte contadores vacios o no numericos en None."""
        try:
            if value is None or value == "":
                return None
            return int(value)
        except Exception:
            return None

    return to_int_or_none(input_tokens), to_int_or_none(output_tokens), to_int_or_none(total_tokens)


def dumps_usage(usage: Dict[str, Any]) -> str:
    """Serializa el uso bruto para conservar informacion especifica del proveedor."""
    return json.dumps(usage, ensure_ascii=False)
