# Valida la coherencia interna de CSV de clasificacion fragmentada.
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from experiment_modules.classification_common import EXPECTED_LABELS


@dataclass
class ValidationIssue:
    """Representa un problema encontrado en un CSV clasificado."""

    level: str
    message: str
    row_index: int | None = None
    row_id: str | None = None


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def validate_probability_columns(
    df: pd.DataFrame,
    prefix: str,
    label_col: str,
    confidence_col: str,
    tolerance: float = 0.02,
) -> List[ValidationIssue]:
    """Comprueba rango, suma, etiqueta argmax y confianza."""
    issues: List[ValidationIssue] = []
    prob_cols = [f"{prefix}{label}" for label in EXPECTED_LABELS]
    missing = [col for col in prob_cols if col not in df.columns]
    if missing:
        return [ValidationIssue("error", f"Faltan columnas de probabilidad: {missing}")]

    probs = df[prob_cols].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = probs.isna().any(axis=1)
    for row_index in invalid_numeric[invalid_numeric].index.tolist()[:10]:
        issues.append(
            ValidationIssue(
                "error",
                f"Hay probabilidades no numericas en {prefix}",
                int(row_index),
                str(df.loc[row_index].get("response_id", "")),
            )
        )

    out_of_range_rows = ((probs < -1e-9) | (probs > 1 + 1e-9)).any(axis=1)
    for row_index in out_of_range_rows[out_of_range_rows].index.tolist()[:10]:
        issues.append(
            ValidationIssue(
                "error",
                "Probabilidades fuera de [0,1]",
                int(row_index),
                str(df.loc[row_index].get("response_id", "")),
            )
        )

    bad_sum_rows = (probs.sum(axis=1) - 1).abs() > tolerance
    for row_index in bad_sum_rows[bad_sum_rows].index.tolist()[:10]:
        issues.append(
            ValidationIssue(
                "error",
                f"Suma de probabilidades != 1: {probs.loc[row_index].sum():.6f}",
                int(row_index),
                str(df.loc[row_index].get("response_id", "")),
            )
        )

    if label_col in df.columns:
        argmax_labels = probs.idxmax(axis=1).str.removeprefix(prefix)
        mismatch_rows = df[label_col].astype(str) != argmax_labels
        for row_index in mismatch_rows[mismatch_rows].index.tolist()[:10]:
            issues.append(
                ValidationIssue(
                    "error",
                    (
                        f"Etiqueta argmax incoherente: {df.loc[row_index, label_col]} "
                        f"!= {argmax_labels.loc[row_index]}"
                    ),
                    int(row_index),
                    str(df.loc[row_index].get("response_id", "")),
                )
            )
    else:
        issues.append(ValidationIssue("error", f"Falta columna de etiqueta: {label_col}"))

    if confidence_col in df.columns:
        max_probs = probs.max(axis=1)
        confidence = _numeric(df, confidence_col)
        invalid_confidence = confidence.isna()
        for row_index in invalid_confidence[invalid_confidence].index.tolist()[:10]:
            issues.append(
                ValidationIssue(
                    "error",
                    f"Confianza no numerica en {confidence_col}",
                    int(row_index),
                    str(df.loc[row_index].get("response_id", "")),
                )
            )
        mismatch_rows = (confidence - max_probs).abs() > tolerance
        for row_index in mismatch_rows[mismatch_rows].index.tolist()[:10]:
            issues.append(
                ValidationIssue(
                    "error",
                    (
                        f"Confianza no coincide con probabilidad maxima: "
                        f"{confidence.loc[row_index]} != {max_probs.loc[row_index]}"
                    ),
                    int(row_index),
                    str(df.loc[row_index].get("response_id", "")),
                )
            )
    else:
        issues.append(ValidationIssue("error", f"Falta columna de confianza: {confidence_col}"))

    return issues


def validate_score_ranges(
    df: pd.DataFrame,
    score_col: str,
    polarization_col: str,
) -> List[ValidationIssue]:
    """Comprueba rangos de sesgo continuo y polarizacion."""
    issues: List[ValidationIssue] = []
    if score_col not in df.columns:
        issues.append(ValidationIssue("error", f"Falta columna de sesgo: {score_col}"))
    else:
        score = _numeric(df, score_col)
        bad_score = score.isna() | (score < -2) | (score > 2)
        for row_index in bad_score[bad_score].index.tolist()[:10]:
            issues.append(
                ValidationIssue(
                    "error",
                    f"Sesgo fuera de [-2,2]: {score.loc[row_index]}",
                    int(row_index),
                    str(df.loc[row_index].get("response_id", "")),
                )
            )

    if polarization_col not in df.columns:
        issues.append(ValidationIssue("error", f"Falta columna de polarizacion: {polarization_col}"))
    else:
        polarization = _numeric(df, polarization_col)
        bad_polarization = polarization.isna() | (polarization < 0) | (polarization > 2)
        for row_index in bad_polarization[bad_polarization].index.tolist()[:10]:
            issues.append(
                ValidationIssue(
                    "error",
                    f"Polarizacion fuera de [0,2]: {polarization.loc[row_index]}",
                    int(row_index),
                    str(df.loc[row_index].get("response_id", "")),
                )
            )

    return issues


def validate_no_duplicates(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> List[ValidationIssue]:
    """Comprueba ausencia de duplicados en columnas identificadoras."""
    issues: List[ValidationIssue] = []
    for column in columns:
        if column not in df.columns:
            continue
        duplicated = df[column].duplicated(keep=False)
        for row_index in duplicated[duplicated].index.tolist()[:10]:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{column} duplicado: {df.loc[row_index, column]}",
                    int(row_index),
                    str(df.loc[row_index].get("response_id", "")),
                )
            )
    return issues


def validate_fragmented_classification_df(df: pd.DataFrame) -> List[ValidationIssue]:
    """Valida el CSV principal de clasificacion fragmentada."""
    issues: List[ValidationIssue] = []
    issues.extend(
        validate_probability_columns(
            df,
            prefix="fragment_prob_",
            label_col="fragment_prob_argmax_ideology_5class",
            confidence_col="fragment_prob_argmax_confidence",
        )
    )
    issues.extend(
        validate_score_ranges(
            df,
            score_col="fragment_weighted_ideology_score",
            polarization_col="fragment_weighted_polarization_score",
        )
    )
    issues.extend(validate_no_duplicates(df, ["response_id", "prompt_id"]))
    return issues
