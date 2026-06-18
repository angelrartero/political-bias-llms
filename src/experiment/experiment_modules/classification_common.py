# Contiene utilidades comunes para cargar modelos y clasificar textos.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiment_modules.paths import RESULTS_DIR


IDEOLOGY_SCORE_5CLASS = {
    "far_left": -2,
    "left": -1,
    "center": 0,
    "right": 1,
    "far_right": 2,
}

EXPECTED_LABELS = list(IDEOLOGY_SCORE_5CLASS.keys())

SCORE_TO_LABEL = {
    -2: "far_left",
    -1: "left",
    0: "center",
    1: "right",
    2: "far_right",
}


def normalize_label(label: str) -> str:
    """Normaliza etiquetas para que coincidan con los nombres internos."""
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def get_device(device_setting: str) -> torch.device:
    """Elige CPU/GPU segun la configuracion indicada."""
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def ensure_parent_dir(path: str | Path) -> None:
    """Crea la carpeta padre de un archivo de salida si no existe."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_csv_input(path: str, delimiter: str) -> pd.DataFrame:
    """Lee un CSV de respuestas y valida que tenga la columna obligatoria."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {path}")

    df = pd.read_csv(path, sep=delimiter, dtype=str).fillna("")

    if "response" not in df.columns:
        raise ValueError("El CSV no contiene la columna obligatoria 'response'.")

    return df


def save_output(df: pd.DataFrame, path: str, delimiter: str) -> None:
    """Guarda un dataframe como CSV usando el delimitador del experimento."""
    ensure_parent_dir(path)
    df.to_csv(path, sep=delimiter, index=False, encoding="utf-8")
    print(f"\nCSV guardado en: {path}")


def load_label_mapping_from_json(model_path: str) -> Optional[Dict[int, str]]:
    """Busca y carga el mapeo id -> etiqueta asociado al modelo."""
    model_dir = Path(model_path)

    # Se prueban varias ubicaciones porque el mapeo puede estar junto al modelo,
    # en la carpeta padre o en results/, segun como se entreno/guardo.
    candidate_paths = [
        model_dir / "label_mapping.json",
        model_dir.parent / "label_mapping.json",
        RESULTS_DIR / model_dir.parent.name / "label_mapping.json",
    ]

    mapping_path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if mapping_path is None:
        return None

    print(f"Usando mapeo de etiquetas desde: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id2label = None

    # Acepta varios formatos habituales de serializacion de etiquetas.
    if isinstance(data, dict) and "id2label" in data:
        id2label = data["id2label"]
    elif isinstance(data, dict) and "label2id" in data:
        id2label = {str(v): k for k, v in data["label2id"].items()}
    elif isinstance(data, dict) and "label_mapping" in data:
        label_mapping = data["label_mapping"]
        if all(not str(k).isdigit() for k in label_mapping.keys()):
            id2label = {str(v): k for k, v in label_mapping.items()}
        elif all(str(k).isdigit() for k in label_mapping.keys()):
            id2label = label_mapping
    elif isinstance(data, dict) and all(str(k).isdigit() for k in data.keys()):
        id2label = data

    if id2label is None:
        return None

    return {int(key): normalize_label(value) for key, value in id2label.items()}


def load_model_and_tokenizer(
    model_path: str,
    device: torch.device,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, Dict[int, str]]:
    """Carga tokenizer, modelo Hugging Face y mapeo de etiquetas."""
    model_dir = Path(model_path)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta del modelo: {model_path}\n"
            "Configura correctamente la ruta al modelo entrenado."
        )

    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"La carpeta existe, pero no contiene config.json: {model_path}\n"
            "Asegurate de apuntar a la carpeta exacta del modelo Hugging Face, "
            "normalmente la subcarpeta best_model/."
        )

    if not (model_dir / "model.safetensors").exists() and not (model_dir / "pytorch_model.bin").exists():
        print(
            "\n[WARN] No se ha encontrado model.safetensors ni pytorch_model.bin "
            f"en: {model_path}\n"
            "Transformers intentara cargar el modelo igualmente."
        )

    print(f"\nCargando tokenizer desde: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    print(f"Cargando modelo desde: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()

    id2label_from_json = load_label_mapping_from_json(model_path)
    if id2label_from_json:
        id2label = id2label_from_json
        print("Usando label_mapping.json para id2label.")
    else:
        print("No se encontro label_mapping.json. Usando model.config.id2label.")
        id2label = {int(k): normalize_label(v) for k, v in model.config.id2label.items()}

    print("\nMapeo de etiquetas cargado:")
    for idx in sorted(id2label.keys()):
        print(f"  {idx}: {id2label[idx]}")

    labels = set(id2label.values())
    expected = set(EXPECTED_LABELS)
    # Las advertencias ayudan a detectar modelos entrenados con otro esquema.
    if not labels.issubset(expected):
        print("\n[WARN] Hay etiquetas que no coinciden con las 5 esperadas:")
        print(f"  Etiquetas del modelo: {sorted(labels)}")
        print(f"  Etiquetas esperadas:  {sorted(expected)}")

    missing_expected = expected - labels
    if missing_expected:
        print("\n[WARN] Faltan algunas etiquetas esperadas en el modelo:")
        print(f"  Faltan: {sorted(missing_expected)}")

    return tokenizer, model, id2label


def classify_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    id2label: Dict[int, str],
    device: torch.device,
    max_length: int,
) -> List[Dict[str, Any]]:
    """Clasifica un batch y devuelve etiqueta, score y probabilidades por clase."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)

    results = []

    for row_probs in probs.cpu():
        # La etiqueta principal es el argmax, pero se conservan todas las
        # probabilidades para resumenes y agregaciones posteriores.
        pred_id = int(torch.argmax(row_probs).item())
        predicted_label = normalize_label(id2label[pred_id])
        confidence = float(row_probs[pred_id].item())
        ideology_score = IDEOLOGY_SCORE_5CLASS.get(predicted_label)
        polarization_score = abs(ideology_score) if ideology_score is not None else None

        row_result = {
            "predicted_ideology_5class": predicted_label,
            "confidence": confidence,
            "ideology_score": ideology_score,
            "polarization_score": polarization_score,
        }

        for idx, label in id2label.items():
            normalized = normalize_label(label)
            row_result[f"prob_{normalized}"] = float(row_probs[int(idx)].item())

        for label in EXPECTED_LABELS:
            # Rellena con 0.0 cualquier clase esperada que no aparezca en id2label.
            row_result.setdefault(f"prob_{label}", 0.0)

        results.append(row_result)

    return results
