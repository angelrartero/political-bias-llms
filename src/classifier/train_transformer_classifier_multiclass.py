from pathlib import Path
import argparse
import math
import json
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning robusto para clasificación multiclase parlamentaria."
    )

    # Datos
    parser.add_argument(
        "--train_path",
        type=str,
        default=str(ROOT / "data" / "processed" / "parliament_multiclass" / "train_split.csv"),
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=str(ROOT / "data" / "processed" / "parliament_multiclass" / "val_split.csv"),
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=str(ROOT / "data" / "processed" / "parliament_multiclass" / "test_split.csv"),
    )
    parser.add_argument("--text_col", type=str, default="text_segment")
    parser.add_argument("--label_col", type=str, default="political_group")

    # Modelo y salidas
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)

    # Tokenización
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="auto",
        choices=["auto", "roberta"],
        help="Tipo de tokenizer a usar.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "roberta"],
        help="Tipo de arquitectura para cargar el modelo.",
    )

    # Entrenamiento
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    # Opciones robustas
    parser.set_defaults(weighted_loss=True)
    parser.add_argument(
        "--no_weighted_loss",
        dest="weighted_loss",
        action="store_false",
        help="Desactiva weighted loss.",
    )

    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Desactiva gradient checkpointing.",
    )

    # Mixed precision
    parser.add_argument("--fp16", action="store_true", help="Fuerza fp16.")
    parser.add_argument("--bf16", action="store_true", help="Fuerza bf16.")

    # Para pruebas rápidas
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    # Reanudar
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


def load_dataframe(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {text_col, label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en {path}: {sorted(missing)}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    df = df[df[text_col] != ""].copy()
    df = df[df[label_col] != ""].copy()
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"El dataframe cargado desde {path} quedó vacío tras la limpieza.")

    return df


def maybe_subsample(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def build_label_mapping(train_df: pd.DataFrame, label_col: str) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(train_df[label_col].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def compute_class_weights(train_df: pd.DataFrame, label_col: str, label2id: dict[str, int]) -> torch.Tensor:
    counts = train_df[label_col].value_counts()
    num_classes = len(label2id)
    total = len(train_df)

    weights = []
    for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
        count = counts[label]
        weight = total / (num_classes * count)
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor


def dataframe_to_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    label2id: dict[str, int],
) -> Dataset:
    unknown_labels = sorted(set(df[label_col]) - set(label2id.keys()))
    if unknown_labels:
        raise ValueError(f"Etiquetas desconocidas encontradas: {unknown_labels}")

    df = df.copy()
    df["labels"] = df[label_col].map(label2id).astype(int)
    df = df.rename(columns={text_col: "text"})

    return Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    def to_python(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_python(v) for v in obj]
        if isinstance(obj, tuple):
            return [to_python(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python(data), f, indent=4, ensure_ascii=False)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig_w = max(10, len(class_names) * 0.9)
    fig_h = max(8, len(class_names) * 0.75)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicción",
        ylabel="Etiqueta real",
        title="Matriz de confusión",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_precision(args: argparse.Namespace) -> tuple[bool, bool]:
    if args.fp16 and args.bf16:
        raise ValueError("No puedes activar fp16 y bf16 a la vez.")

    if args.fp16:
        return True, False
    if args.bf16:
        return False, True

    if not torch.cuda.is_available():
        return False, False

    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return False, True

    return True, False


def load_tokenizer(args: argparse.Namespace):
    if args.tokenizer_type == "roberta":
        return RobertaTokenizer.from_pretrained(args.model_name)

    try:
        return AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    except Exception:
        # Algunos repositorios no pueden construir el tokenizador rápido sin dependencias opcionales.
        try:
            return AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        except Exception:
            try:
                return RobertaTokenizer.from_pretrained(args.model_name)
            except Exception as exc:
                raise RuntimeError(
                    "No se pudo cargar el tokenizer. Prueba con --tokenizer_type roberta "
                    "o instala dependencias opcionales: pip install sentencepiece tiktoken"
                ) from exc


def load_sequence_classifier_model(
    args: argparse.Namespace,
    label2id: dict[str, int],
    id2label: dict[int, str],
):
    common_kwargs = {
        "num_labels": len(label2id),
        "id2label": id2label,
        "label2id": label2id,
    }

    if args.model_type == "roberta":
        return RobertaForSequenceClassification.from_pretrained(
            args.model_name,
            **common_kwargs,
        )

    return AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        **common_kwargs,
    )


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CARGA DE DATOS")
    print("=" * 70)

    train_df = load_dataframe(args.train_path, args.text_col, args.label_col)
    val_df = load_dataframe(args.val_path, args.text_col, args.label_col)
    test_df = load_dataframe(args.test_path, args.text_col, args.label_col)

    train_df = maybe_subsample(train_df, args.max_train_samples, args.seed)
    val_df = maybe_subsample(val_df, args.max_val_samples, args.seed)
    test_df = maybe_subsample(test_df, args.max_test_samples, args.seed)

    label2id, id2label = build_label_mapping(train_df, args.label_col)
    class_names = [id2label[i] for i in range(len(id2label))]

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"Número de clases: {len(label2id)}")
    print("Distribución train:")
    print(train_df[args.label_col].value_counts())

    class_weights = compute_class_weights(train_df, args.label_col, label2id) if args.weighted_loss else None
    if class_weights is not None:
        print("\nClass weights:")
        for idx, weight in enumerate(class_weights.tolist()):
            print(f"  {id2label[idx]}: {weight:.4f}")

    dataset = DatasetDict(
        {
            "train": dataframe_to_dataset(train_df, args.text_col, args.label_col, label2id),
            "validation": dataframe_to_dataset(val_df, args.text_col, args.label_col, label2id),
            "test": dataframe_to_dataset(test_df, args.text_col, args.label_col, label2id),
        }
    )

    print("\n" + "=" * 70)
    print("TOKENIZER Y MODELO")
    print("=" * 70)

    tokenizer = load_tokenizer(args)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("El tokenizer no tiene pad_token, eos_token ni unk_token.")

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizando dataset",
    )

    model = load_sequence_classifier_model(args, label2id, id2label)

    model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    fp16, bf16 = resolve_precision(args)
    warmup_steps = math.ceil(
        math.ceil(len(tokenized["train"]) / max(1, args.train_batch_size))
        * args.num_train_epochs
        * args.warmup_ratio
    )

    print(f"fp16: {fp16}")
    print(f"bf16: {bf16}")
    print(f"gradient_checkpointing: {args.gradient_checkpointing}")
    print(f"weighted_loss: {args.weighted_loss}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
    )

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO")
    print("=" * 70)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    print("\n" + "=" * 70)
    print("EVALUACIÓN FINAL EN TEST")
    print("=" * 70)

    pred_output = trainer.predict(tokenized["test"])
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = np.array(tokenized["test"]["labels"])

    metrics = compute_metrics((pred_output.predictions, y_true))

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=results_dir / "confusion_matrix.png",
    )

    metrics_payload = {
        **metrics,
        "model_name": args.model_name,
        "num_labels": len(label2id),
        "label_mapping": label2id,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "classification_report": report_dict,
    }

    config_payload = {
        "model_name": args.model_name,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "test_path": args.test_path,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "max_length": args.max_length,
        "tokenizer_type": args.tokenizer_type,
        "model_type": args.model_type,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "weighted_loss": args.weighted_loss,
        "gradient_checkpointing": args.gradient_checkpointing,
        "warmup_steps": warmup_steps,
        "fp16": fp16,
        "bf16": bf16,
        "label_mapping": label2id,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "train_distribution": train_df[args.label_col].value_counts().to_dict(),
        "val_distribution": val_df[args.label_col].value_counts().to_dict(),
        "test_distribution": test_df[args.label_col].value_counts().to_dict(),
    }

    save_json(metrics_payload, results_dir / "metrics.json")
    save_json(config_payload, results_dir / "config.json")
    save_json({"label_mapping": label2id}, results_dir / "label_mapping.json")

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(report_text)
    print(f"\nMejor checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Modelo final guardado en: {best_model_dir}")
    print(f"Resultados guardados en: {results_dir}")


if __name__ == "__main__":
    main()