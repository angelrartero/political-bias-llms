# Ejecuta la clasificacion ideologica de respuestas completas de LLM.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiment_modules.classification_common import (
    classify_batch,
    get_device,
    load_model_and_tokenizer,
    read_csv_input,
    save_output,
)
from experiment_modules.classification_paths import build_classified_output_csv, build_summary_json
from experiment_modules.classification_summary import (
    build_classification_summary,
    print_basic_classification_summary,
    save_summary_json,
)
from experiment_modules.paths import CLASSIFICATIONS_DIR, MODELS_DIR, RESPONSES_DIR


DEFAULT_INPUT_CSV = str(
    RESPONSES_DIR / "debug" / "debug_claude_haiku_4_5_5prompts_spanish_context_unparrafo.csv"
)
DEFAULT_OUTPUT_CSV = str(
    CLASSIFICATIONS_DIR / "debug" / "classified_claude_haiku_4_5_5prompts_spanish_context_unparrafo.csv"
)
DEFAULT_MODEL_PATH = str(
    MODELS_DIR / "mrbert-v2-5class-27-04" / "best_model"
)
DEFAULT_SUMMARY_JSON = str(Path(DEFAULT_OUTPUT_CSV).with_suffix(".summary.json"))


@dataclass
class ClassifierSettings:
    """Parametros para clasificar respuestas completas sin fragmentacion."""

    input_csv: str = DEFAULT_INPUT_CSV
    output_csv: str = DEFAULT_OUTPUT_CSV
    summary_json: str = DEFAULT_SUMMARY_JSON
    model_path: str = DEFAULT_MODEL_PATH
    delimiter: str = ";"
    text_column: str = "response"
    status_column: str = "status"
    success_only: bool = True
    batch_size: int = 8
    max_length: int = 512
    device: str = "auto"


def classify_dataframe(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    id2label: Dict[int, str],
    settings: ClassifierSettings,
    device: torch.device,
) -> pd.DataFrame:
    """Filtra respuestas validas, las clasifica por batches y une resultados."""
    if settings.text_column not in df.columns:
        raise ValueError(f"No existe la columna de texto: {settings.text_column}")

    work_df = df.copy()

    if settings.success_only and settings.status_column in work_df.columns:
        # Normalmente solo interesan respuestas generadas correctamente por la API.
        before = len(work_df)
        work_df = work_df[work_df[settings.status_column] == "success"].copy()
        after = len(work_df)
        print(f"\nFiltrando status='success': {before} -> {after} filas")

    work_df = work_df[work_df[settings.text_column].astype(str).str.strip() != ""].copy()
    texts = work_df[settings.text_column].astype(str).tolist()

    print(f"\nClasificando {len(texts)} respuestas...")
    print(f"Device: {device}")
    print(f"Batch size: {settings.batch_size}")
    print(f"Max length: {settings.max_length}")

    all_predictions: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(texts), settings.batch_size), desc="Clasificando"):
        # Se reutiliza classify_batch para mantener la misma logica que el
        # clasificador por fragmentos.
        batch_predictions = classify_batch(
            texts=texts[start:start + settings.batch_size],
            tokenizer=tokenizer,
            model=model,
            id2label=id2label,
            device=device,
            max_length=settings.max_length,
        )
        all_predictions.extend(batch_predictions)

    pred_df = pd.DataFrame(all_predictions)
    return pd.concat([work_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)


def run_classification(settings: ClassifierSettings) -> None:
    """Ejecuta lectura, carga de modelo, clasificacion, guardado y resumen."""
    if settings.output_csv == DEFAULT_OUTPUT_CSV:
        # Si se usa la salida por defecto, se deriva una ruta segun el CSV elegido.
        settings.output_csv = str(build_classified_output_csv(settings.input_csv))

    if settings.summary_json == DEFAULT_SUMMARY_JSON:
        settings.summary_json = str(build_summary_json(settings.output_csv))

    device = get_device(settings.device)
    df = read_csv_input(settings.input_csv, settings.delimiter)

    tokenizer, model, id2label = load_model_and_tokenizer(settings.model_path, device)

    classified_df = classify_dataframe(
        df=df,
        tokenizer=tokenizer,
        model=model,
        id2label=id2label,
        settings=settings,
        device=device,
    )

    save_output(classified_df, settings.output_csv, settings.delimiter)

    summary = build_classification_summary(
        classified_df=classified_df,
        original_df=df,
        settings=settings,
        id2label=id2label,
        device=device,
    )
    save_summary_json(summary, settings.summary_json)
    print_basic_classification_summary(classified_df)
