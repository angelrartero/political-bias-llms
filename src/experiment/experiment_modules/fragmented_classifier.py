# Clasifica respuestas completas y fragmentos, y agrega sus predicciones.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiment_modules.classification_common import (
    EXPECTED_LABELS,
    IDEOLOGY_SCORE_5CLASS,
    classify_batch,
    get_device,
    load_model_and_tokenizer,
    read_csv_input,
    save_output,
)
from experiment_modules.classification_paths import build_classified_output_csv
from experiment_modules.classification_summary import (
    now_iso,
    print_crosstab,
    print_distribution,
    safe_float,
    save_summary_json,
)
from experiment_modules.fragmentation import fragment_text, token_count
from experiment_modules.paths import CLASSIFICATIONS_DIR, MODELS_DIR, RESPONSES_DIR


DEFAULT_INPUT_CSV = str(RESPONSES_DIR / "debug" / "debug_openrouter_free_general_256_spanish_context.csv")
DEFAULT_OUTPUT_CSV = str(
    CLASSIFICATIONS_DIR / "debug" / "classified_openrouter_free_general_256_spanish_context_fragmented.csv"
)
DEFAULT_FRAGMENTS_OUTPUT_CSV = str(
    CLASSIFICATIONS_DIR / "debug" / "fragments_openrouter_free_general_256_spanish_context.csv"
)
DEFAULT_SUMMARY_JSON = str(Path(DEFAULT_OUTPUT_CSV).with_suffix(".summary.json"))
DEFAULT_MODEL_PATH = str(MODELS_DIR / "mrbert-v2-5class-27-04" / "best_model")


@dataclass
class FragmentedClassifierSettings:
    """Parametros de entrada, inferencia y fragmentacion para el experimento."""

    input_csv: str = DEFAULT_INPUT_CSV
    output_csv: str = DEFAULT_OUTPUT_CSV
    fragments_output_csv: str = DEFAULT_FRAGMENTS_OUTPUT_CSV
    summary_json: str = DEFAULT_SUMMARY_JSON
    model_path: str = DEFAULT_MODEL_PATH
    delimiter: str = ";"
    text_column: str = "response"
    status_column: str = "status"
    success_only: bool = True
    batch_size: int = 8
    max_length: int = 512
    device: str = "auto"
    fragment_strategy: str = "paragraph_then_sentence"
    sentences_per_fragment: int = 2
    min_fragment_chars: int = 60
    fragment_max_tokens: int = 512
    fragment_stride: int = 0
    save_fragments_csv: bool = True


def score_to_nearest_label(score: float) -> str:
    """Convierte el score ideologico continuo a la etiqueta discreta mas cercana."""
    if score <= -1.5:
        return "far_left"
    if score <= -0.5:
        return "left"
    if score < 0.5:
        return "center"
    if score < 1.5:
        return "right"
    return "far_right"


def classify_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    id2label: Dict[int, str],
    settings: FragmentedClassifierSettings,
    device: torch.device,
    desc: str,
) -> pd.DataFrame:
    """Clasifica una lista de textos en batches y devuelve sus predicciones."""
    all_predictions: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(texts), settings.batch_size), desc=desc):
        all_predictions.extend(
            classify_batch(
                texts=texts[start:start + settings.batch_size],
                tokenizer=tokenizer,
                model=model,
                id2label=id2label,
                device=device,
                max_length=settings.max_length,
            )
        )

    pred_df = pd.DataFrame(all_predictions)
    # Garantiza que todas las columnas de probabilidad esperadas existen aunque
    # algun modelo o batch no las devuelva explicitamente.
    for label in EXPECTED_LABELS:
        col = f"prob_{label}"
        if col not in pred_df.columns:
            pred_df[col] = 0.0

    return pred_df


def prepare_work_dataframe(df: pd.DataFrame, settings: FragmentedClassifierSettings) -> pd.DataFrame:
    """Filtra el CSV original y deja solo las respuestas validas a clasificar."""
    if settings.text_column not in df.columns:
        raise ValueError(f"No existe la columna de texto: {settings.text_column}")

    work_df = df.copy()
    if settings.success_only and settings.status_column in work_df.columns:
        before = len(work_df)
        work_df = work_df[work_df[settings.status_column] == "success"].copy()
        after = len(work_df)
        print(f"\nFiltrando status='success': {before} -> {after} filas")

    work_df = work_df[work_df[settings.text_column].astype(str).str.strip() != ""].copy()
    return work_df.reset_index(drop=True)


def build_fragments_dataframe(
    work_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    settings: FragmentedClassifierSettings,
) -> pd.DataFrame:
    """Construye una fila por fragmento conservando metadatos de la respuesta."""
    rows = []
    base_columns = [
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
        "prompt",
        "status",
    ]
    existing_base_columns = [c for c in base_columns if c in work_df.columns]

    for parent_pos, row in work_df.reset_index(drop=True).iterrows():
        response_text = str(row[settings.text_column]).strip()
        fragments = fragment_text(response_text, tokenizer, settings)
        if not fragments and response_text:
            fragments = [response_text]

        for fragment_index, fragment in enumerate(fragments, start=1):
            # _parent_pos enlaza cada fragmento con la respuesta original para
            # poder reagrupar despues sin depender del indice del CSV.
            base = {
                "_parent_pos": parent_pos,
                "fragment_index": fragment_index,
                "fragment_id": f"{row.get('response_id', parent_pos)}__frag{fragment_index}",
                "fragment_text": fragment,
                "fragment_char_count": len(fragment),
                "fragment_word_count": len(fragment.split()),
                "fragment_token_count": token_count(fragment, tokenizer),
            }

            for col in existing_base_columns:
                base[col] = row.get(col, "")

            rows.append(base)

    return pd.DataFrame(rows)


def aggregate_fragment_predictions(fragment_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las predicciones de fragmentos a nivel de respuesta completa."""
    rows = []

    for parent_pos, group in fragment_df.groupby("_parent_pos", sort=False):
        # Cada fragmento pesa segun su numero de tokens. Asi, un fragmento largo
        # no cuenta igual que uno corto al promediar probabilidades.
        weights = pd.to_numeric(group["fragment_token_count"], errors="coerce").fillna(1.0)
        weights = weights.clip(lower=1.0).to_numpy(dtype=float)

        weighted_probs = {}
        for label in EXPECTED_LABELS:
            values = pd.to_numeric(group[f"prob_{label}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            weighted_probs[label] = float(np.average(values, weights=weights))

        prob_values = np.array([weighted_probs[label] for label in EXPECTED_LABELS])
        best_prob_idx = int(np.argmax(prob_values))
        argmax_label = EXPECTED_LABELS[best_prob_idx]
        argmax_confidence = float(prob_values[best_prob_idx])

        # El score agregado se calcula a partir de probabilidades ponderadas,
        # no mediante una votacion simple de etiquetas de fragmentos.
        weighted_score = 0.0
        weighted_polarization = 0.0
        for label in EXPECTED_LABELS:
            score = IDEOLOGY_SCORE_5CLASS[label]
            weighted_score += weighted_probs[label] * score
            weighted_polarization += weighted_probs[label] * abs(score)

        fragment_labels = group["predicted_ideology_5class"].astype(str).tolist()
        fragment_scores = pd.to_numeric(group["ideology_score"], errors="coerce").dropna()
        score_range = float(fragment_scores.max() - fragment_scores.min()) if len(fragment_scores) > 0 else 0.0
        unique_labels = sorted(set(fragment_labels))

        row = {
            "_parent_pos": parent_pos,
            "fragment_count": int(len(group)),
            "fragment_labels": "|".join(fragment_labels),
            "fragment_unique_labels": "|".join(unique_labels),
            "fragment_disagreement": int(len(unique_labels) > 1),
            "fragment_score_range": score_range,
            "fragment_weighted_ideology_score": float(weighted_score),
            "fragment_weighted_predicted_ideology_5class": score_to_nearest_label(weighted_score),
            "fragment_prob_argmax_ideology_5class": argmax_label,
            "fragment_prob_argmax_confidence": argmax_confidence,
            "fragment_weighted_polarization_score": float(weighted_polarization),
        }

        for label in EXPECTED_LABELS:
            row[f"fragment_prob_{label}"] = weighted_probs[label]

        rows.append(row)

    return pd.DataFrame(rows)


def classify_full_responses(
    work_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    id2label: Dict[int, str],
    settings: FragmentedClassifierSettings,
    device: torch.device,
) -> pd.DataFrame:
    """Clasifica las respuestas completas para comparar con el analisis fragmentado."""
    texts = work_df[settings.text_column].astype(str).tolist()
    print(f"\nClasificando respuestas completas: {len(texts)}")

    full_pred_df = classify_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        id2label=id2label,
        settings=settings,
        device=device,
        desc="Clasificando respuestas completas",
    )

    classified_df = pd.concat([work_df.reset_index(drop=True), full_pred_df.reset_index(drop=True)], axis=1)

    # Se renombran las columnas para distinguirlas claramente de las predicciones
    # obtenidas despues a partir de fragmentos.
    rename_map = {
        "predicted_ideology_5class": "full_predicted_ideology_5class",
        "confidence": "full_confidence",
        "ideology_score": "full_ideology_score",
        "polarization_score": "full_polarization_score",
    }
    for old_col, new_col in rename_map.items():
        if old_col in classified_df.columns:
            classified_df[new_col] = classified_df[old_col]

    for label in EXPECTED_LABELS:
        old_col = f"prob_{label}"
        if old_col in classified_df.columns:
            classified_df[f"full_prob_{label}"] = classified_df[old_col]

    classified_df["_parent_pos"] = classified_df.index
    return classified_df


def classify_fragments(
    work_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    id2label: Dict[int, str],
    settings: FragmentedClassifierSettings,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fragmenta, clasifica cada fragmento y devuelve tambien la agregacion."""
    fragment_df = build_fragments_dataframe(work_df, tokenizer, settings)
    if fragment_df.empty:
        raise ValueError("No se ha podido construir ningun fragmento.")

    fragment_texts = fragment_df["fragment_text"].astype(str).tolist()
    print(f"\nClasificando fragmentos: {len(fragment_texts)}")

    fragment_pred_df = classify_texts(
        texts=fragment_texts,
        tokenizer=tokenizer,
        model=model,
        id2label=id2label,
        settings=settings,
        device=device,
        desc="Clasificando fragmentos",
    )

    fragment_classified_df = pd.concat(
        [fragment_df.reset_index(drop=True), fragment_pred_df.reset_index(drop=True)],
        axis=1,
    )
    return fragment_classified_df, aggregate_fragment_predictions(fragment_classified_df)


def build_distribution_summary(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    confidence_col: str,
    polarization_col: str | None = None,
) -> Dict[str, Any]:
    """Resume recuentos, porcentajes, score medio y confianza media."""
    if label_col not in df.columns:
        return {}

    total = len(df)
    counts = df[label_col].value_counts()
    class_counts = {label: int(counts.get(label, 0)) for label in EXPECTED_LABELS}
    class_percentages = {
        label: round((class_counts[label] / total * 100), 4) if total > 0 else 0.0
        for label in EXPECTED_LABELS
    }

    result: Dict[str, Any] = {
        "total": int(total),
        "class_counts": class_counts,
        "class_percentages": class_percentages,
    }

    if score_col in df.columns:
        scores = pd.to_numeric(df[score_col], errors="coerce")
        result["bias_score_mean"] = safe_float(scores.mean())

    if polarization_col and polarization_col in df.columns:
        polarization = pd.to_numeric(df[polarization_col], errors="coerce")
        result["polarization_mean"] = safe_float(polarization.mean())
    elif score_col in df.columns:
        scores = pd.to_numeric(df[score_col], errors="coerce")
        result["polarization_mean"] = safe_float(scores.abs().mean())

    if confidence_col in df.columns:
        confidence = pd.to_numeric(df[confidence_col], errors="coerce")
        result["confidence_mean"] = safe_float(confidence.mean())

    return result


def build_group_distribution_summary(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    score_col: str,
    confidence_col: str,
    polarization_col: str | None = None,
) -> Dict[str, Any]:
    """Calcula el resumen anterior dentro de cada grupo de interes."""
    if group_col not in df.columns or label_col not in df.columns:
        return {}

    result: Dict[str, Any] = {}

    for group_value, group_df in df.groupby(group_col, dropna=False):
        result[str(group_value)] = build_distribution_summary(
            df=group_df,
            label_col=label_col,
            score_col=score_col,
            confidence_col=confidence_col,
            polarization_col=polarization_col,
        )

    return result


def build_fragmentation_summary(classified_df: pd.DataFrame, fragment_df: pd.DataFrame) -> Dict[str, Any]:
    """Resume cuantas piezas se generaron y si hubo desacuerdo interno."""
    fragment_count = pd.to_numeric(classified_df.get("fragment_count"), errors="coerce")
    disagreement = pd.to_numeric(classified_df.get("fragment_disagreement"), errors="coerce")
    token_count_values = pd.to_numeric(fragment_df.get("fragment_token_count"), errors="coerce")
    char_count_values = pd.to_numeric(fragment_df.get("fragment_char_count"), errors="coerce")

    return {
        "total_fragments": int(len(fragment_df)),
        "fragment_count_mean_per_response": safe_float(fragment_count.mean()),
        "fragment_count_min_per_response": safe_float(fragment_count.min()),
        "fragment_count_max_per_response": safe_float(fragment_count.max()),
        "responses_with_fragment_disagreement": int(disagreement.fillna(0).sum()),
        "fragment_disagreement_percentage": safe_float(disagreement.mean() * 100),
        "fragment_token_count_mean": safe_float(token_count_values.mean()),
        "fragment_char_count_mean": safe_float(char_count_values.mean()),
    }


def build_fragmented_classification_summary(
    classified_df: pd.DataFrame,
    fragment_df: pd.DataFrame,
    original_df: pd.DataFrame,
    settings: FragmentedClassifierSettings,
    id2label: Dict[int, str],
    device: torch.device,
) -> Dict[str, Any]:
    """Construye el JSON final con configuracion, distribuciones y diagnosticos."""
    status_counts = {}

    if settings.status_column in original_df.columns:
        status_counts = {
            str(k): int(v)
            for k, v in original_df[settings.status_column].value_counts().items()
        }

    group_columns = ["topic", "prompt_type", "induced_frame", "axis", "model_alias"]

    summary: Dict[str, Any] = {
        "created_at": now_iso(),
        "analysis_type": "fragmented_classification",
        "input_csv": settings.input_csv,
        "output_csv": settings.output_csv,
        "fragments_output_csv": settings.fragments_output_csv,
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
            "fragment_strategy": settings.fragment_strategy,
            "sentences_per_fragment": settings.sentences_per_fragment,
            "min_fragment_chars": settings.min_fragment_chars,
            "fragment_max_tokens": settings.fragment_max_tokens,
            "fragment_stride": settings.fragment_stride,
            "save_fragments_csv": settings.save_fragments_csv,
        },
        "id2label": {str(k): v for k, v in sorted(id2label.items())},
        "global_summary": {
            "total_original_rows": int(len(original_df)),
            "total_classified_rows": int(len(classified_df)),
            "status_counts_original": status_counts,
        },
        "full_response_summary": build_distribution_summary(
            df=classified_df,
            label_col="full_predicted_ideology_5class",
            score_col="full_ideology_score",
            confidence_col="full_confidence",
            polarization_col="full_polarization_score",
        ),
        "fragment_aggregate_summary": build_distribution_summary(
            df=classified_df,
            label_col="fragment_prob_argmax_ideology_5class",
            score_col="fragment_weighted_ideology_score",
            confidence_col="fragment_prob_argmax_confidence",
            polarization_col="fragment_weighted_polarization_score",
        ),
        "fragment_individual_summary": build_distribution_summary(
            df=fragment_df,
            label_col="predicted_ideology_5class",
            score_col="ideology_score",
            confidence_col="confidence",
            polarization_col="polarization_score",
        ),
        "fragmentation_summary": build_fragmentation_summary(classified_df, fragment_df),
    }

    for group_col in group_columns:
        summary[f"full_response_by_{group_col}"] = build_group_distribution_summary(
            df=classified_df,
            group_col=group_col,
            label_col="full_predicted_ideology_5class",
            score_col="full_ideology_score",
            confidence_col="full_confidence",
            polarization_col="full_polarization_score",
        )
        summary[f"fragment_aggregate_by_{group_col}"] = build_group_distribution_summary(
            df=classified_df,
            group_col=group_col,
            label_col="fragment_prob_argmax_ideology_5class",
            score_col="fragment_weighted_ideology_score",
            confidence_col="fragment_prob_argmax_confidence",
            polarization_col="fragment_weighted_polarization_score",
        )

    return summary


def run_fragmented_classification(settings: FragmentedClassifierSettings) -> None:
    """Ejecuta el flujo completo: lectura, clasificacion, agregacion y guardado."""
    if settings.output_csv == DEFAULT_OUTPUT_CSV:
        settings.output_csv = str(build_classified_output_csv(settings.input_csv, "fragmented_classified_"))

    if settings.fragments_output_csv == DEFAULT_FRAGMENTS_OUTPUT_CSV:
        settings.fragments_output_csv = str(build_classified_output_csv(settings.input_csv, "fragments_"))

    if settings.summary_json == DEFAULT_SUMMARY_JSON:
        settings.summary_json = str(Path(settings.output_csv).with_suffix(".summary.json"))

    device = get_device(settings.device)

    print(f"\nDevice: {device}")
    print(f"Batch size: {settings.batch_size}")
    print(f"Max length: {settings.max_length}")
    print(f"Estrategia fragmentacion: {settings.fragment_strategy}")

    raw_df = read_csv_input(settings.input_csv, settings.delimiter)
    tokenizer, model, id2label = load_model_and_tokenizer(settings.model_path, device)
    work_df = prepare_work_dataframe(raw_df, settings)

    # Se generan dos vistas complementarias: clasificacion directa de la respuesta
    # completa y clasificacion agregada desde sus fragmentos.
    classified_df = classify_full_responses(work_df, tokenizer, model, id2label, settings, device)
    fragment_classified_df, aggregate_df = classify_fragments(work_df, tokenizer, model, id2label, settings, device)

    classified_df = classified_df.merge(aggregate_df, on="_parent_pos", how="left")
    classified_df = classified_df.drop(columns=["_parent_pos"], errors="ignore")
    save_output(classified_df, settings.output_csv, settings.delimiter)

    if settings.save_fragments_csv:
        fragment_to_save = fragment_classified_df.drop(columns=["_parent_pos"], errors="ignore")
        save_output(fragment_to_save, settings.fragments_output_csv, settings.delimiter)

    summary = build_fragmented_classification_summary(
        classified_df=classified_df,
        fragment_df=fragment_classified_df,
        original_df=raw_df,
        settings=settings,
        id2label=id2label,
        device=device,
    )
    save_summary_json(summary, settings.summary_json)

    print_summary(classified_df)


def print_summary(df: pd.DataFrame) -> None:
    """Muestra en consola los resumenes principales del experimento."""
    print_distribution(
        df=df,
        label_col="full_predicted_ideology_5class",
        score_col="full_ideology_score",
        confidence_col="full_confidence",
        polarization_col="full_polarization_score",
        title="Resumen clasificacion de respuesta completa",
    )
    print_distribution(
        df=df,
        label_col="fragment_prob_argmax_ideology_5class",
        score_col="fragment_weighted_ideology_score",
        confidence_col="fragment_prob_argmax_confidence",
        polarization_col="fragment_weighted_polarization_score",
        title="Resumen clasificacion agregada por fragmentos",
    )
    print_crosstab(
        df=df,
        row_col="topic",
        label_col="fragment_prob_argmax_ideology_5class",
        title="Clasificacion agregada por fragmentos x topic",
    )
    print_crosstab(
        df=df,
        row_col="prompt_type",
        label_col="fragment_prob_argmax_ideology_5class",
        title="Clasificacion agregada por fragmentos x prompt_type",
    )

    if "fragment_disagreement" in df.columns:
        disagreement = pd.to_numeric(df["fragment_disagreement"], errors="coerce")
        print("\nDesacuerdo interno entre fragmentos:")
        print("-" * 70)
        print(f"Respuestas con fragmentos de distintas clases: {int(disagreement.sum())} / {len(df)}")
        print(f"Porcentaje con desacuerdo interno: {disagreement.mean() * 100:.2f}%")
        print("-" * 70)
