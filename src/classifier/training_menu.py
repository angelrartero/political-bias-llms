from pathlib import Path
from datetime import datetime
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]

RESULTS_BASE_DIR = ROOT / "results" / "parliament_multiclass"
MODELS_BASE_DIR = ROOT / "models" / "parliament_multiclass"


MODELS = {
    "1": {
        "display_name": "MarIA RoBERTa",
        "folder_name": "maria_roberta",
        "model_name": "PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy",
        "tokenizer_type": "roberta",
        "model_type": "roberta",
        "learning_rate": "1.5e-5",
        "weight_decay": "0.01",
        "num_train_epochs": "4",
        "train_batch_size": "4",
        "eval_batch_size": "8",
        "gradient_accumulation_steps": "4",
        "dataloader_num_workers": "4",
    },
    "2": {
        "display_name": "MrBERT",
        "folder_name": "mrbert",
        "model_name": "BSC-LT/MrBERT-es",
        "tokenizer_type": "auto",
        "model_type": "auto",
        "learning_rate": "2e-5",
        "weight_decay": "0.01",
        "num_train_epochs": "3",
        "train_batch_size": "8",
        "eval_batch_size": "16",
        "gradient_accumulation_steps": "2",
        "dataloader_num_workers": "2",
    },
}


DATASETS = {
    # ============================================================
    # DATASET ORIGINAL MULTICLASE POR PARTIDO
    # ============================================================
    "1": {
        "display_name": "Original - multiclase por partido",
        "folder_name": "original",
        "label_scheme": "party",
        "base_path": ROOT / "data" / "processed" / "parliament_multiclass",
        "label_col": "political_group",
        "text_col": "text_segment",
    },

    # ============================================================
    # DATASETS TERNARIOS
    # ============================================================
    "2": {
        "display_name": "Ternario v1 - Todo el dataset",
        "folder_name": "v1",
        "label_scheme": "3class",
        "base_path": ROOT / "data" / "processed" / "parliament_ideology" / "v1_full_ideology",
        "label_col": "ideology_label",
        "text_col": "text",
    },
    "3": {
        "display_name": "Ternario v2 - Desde legislatura 13",
        "folder_name": "v2",
        "label_scheme": "3class",
        "base_path": ROOT / "data" / "processed" / "parliament_ideology" / "v2_since_legislature_13_ideology",
        "label_col": "ideology_label",
        "text_col": "text",
    },
    "4": {
        "display_name": "Ternario v3 - Desde legislatura 13 balanceado",
        "folder_name": "v3",
        "label_scheme": "3class",
        "base_path": ROOT / "data" / "processed" / "parliament_ideology" / "v3_since_legislature_13_balanced",
        "label_col": "ideology_label",
        "text_col": "text",
    },
    "5": {
        "display_name": "Ternario v4 - Desde legislatura 13 sin nacionalistas y balanceado",
        "folder_name": "v4",
        "label_scheme": "3class",
        "base_path": ROOT / "data" / "processed" / "parliament_ideology" / "v4_since_legislature_13_no_nationalist_balanced",
        "label_col": "ideology_label",
        "text_col": "text",
    },

    # ============================================================
    # DATASETS DE 5 CLASES
    # ============================================================
    "6": {
        "display_name": "5 clases v1 - Todo el dataset",
        "folder_name": "v1",
        "label_scheme": "5class",
        "base_path": ROOT / "data" / "processed" / "parliament_ideology_5class" / "v1_full_5class",
        "label_col": "ideology_5_label",
        "text_col": "text",
    },
    "7": {
        "display_name": "5 clases v2 - Desde legislatura 13",
        "folder_name": "v2",
        "label_scheme": "5class",
        "base_path": ROOT / "data" / "processed" / "parliament_ideology_5class" / "v2_since_legislature_13_5class",
        "label_col": "ideology_5_label",
        "text_col": "text",
    },
    "8": {
        "display_name": "5 clases v3 - Desde legislatura 13 balanceado",
        "folder_name": "v3",
        "label_scheme": "5class",
        "base_path": ROOT / "data" / "processed" / "parliament_ideology_5class" / "v3_since_legislature_13_5class_balanced",
        "label_col": "ideology_5_label",
        "text_col": "text",
    },
    "9": {
        "display_name": "5 clases v4 - Desde legislatura 13 sin nacionalistas y balanceado",
        "folder_name": "v4",
        "label_scheme": "5class",
        "base_path": ROOT / "data" / "processed" / "parliament_ideology_5class" / "v4_since_legislature_13_5class_no_nationalist_balanced",
        "label_col": "ideology_5_label",
        "text_col": "text",
    },
    "10": {
        "display_name": "6 clases v5 - Desde legislatura 13 + nacionalista + balanceado",
        "folder_name": "v5",
        "label_scheme": "6class",
        "base_path": ROOT / "data" / "processed" / "parliament_ideology_5class" / "v5_since_legislature_13_6class_nationalist_balanced",
        "label_col": "ideology_5_label",
        "text_col": "text",
    },
}


def show_menu(title: str, options: dict) -> str:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for key, value in options.items():
        print(f"{key}. {value['display_name']}")

    print("0. Salir")

    option = input("\nSelecciona una opción: ").strip()

    if option == "0":
        print("Proceso cancelado.")
        sys.exit(0)

    if option not in options:
        print("Opción no válida.")
        sys.exit(1)

    return option


def check_dataset_paths(dataset_config: dict) -> tuple[Path, Path, Path]:
    base_path = dataset_config["base_path"]

    train_path = base_path / "train_split.csv"
    val_path = base_path / "val_split.csv"
    test_path = base_path / "test_split.csv"

    missing = [
        path for path in [train_path, val_path, test_path]
        if not path.exists()
    ]

    if missing:
        print("\nERROR: faltan archivos del dataset:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)

    return train_path, val_path, test_path


def build_experiment_name(model_config: dict, dataset_config: dict) -> str:
    date_suffix = datetime.now().strftime("%d-%m")

    if dataset_config["label_scheme"] == "party":
        return (
            f"{model_config['folder_name']}-"
            f"{dataset_config['folder_name']}-"
            f"{date_suffix}"
        )

    return (
        f"{model_config['folder_name']}-"
        f"{dataset_config['folder_name']}-"
        f"{dataset_config['label_scheme']}-"
        f"{date_suffix}"
    )


def run_training(model_config: dict, dataset_config: dict) -> None:
    train_path, val_path, test_path = check_dataset_paths(dataset_config)

    experiment_name = build_experiment_name(model_config, dataset_config)

    output_dir = MODELS_BASE_DIR / experiment_name
    results_dir = RESULTS_BASE_DIR / experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.parliament_multiclass.train_transformer_classifier_multiclass",

        "--train_path", str(train_path),
        "--val_path", str(val_path),
        "--test_path", str(test_path),

        "--text_col", dataset_config["text_col"],
        "--label_col", dataset_config["label_col"],

        "--model_name", model_config["model_name"],
        "--tokenizer_type", model_config["tokenizer_type"],
        "--model_type", model_config["model_type"],

        "--output_dir", str(output_dir),
        "--results_dir", str(results_dir),

        "--max_length", "512",
        "--learning_rate", model_config["learning_rate"],
        "--weight_decay", model_config["weight_decay"],
        "--num_train_epochs", model_config["num_train_epochs"],

        "--train_batch_size", model_config["train_batch_size"],
        "--eval_batch_size", model_config["eval_batch_size"],
        "--gradient_accumulation_steps", model_config["gradient_accumulation_steps"],

        "--warmup_ratio", "0.1",
        "--lr_scheduler_type", "cosine",
        "--max_grad_norm", "1.0",

        "--logging_steps", "100",
        "--save_total_limit", "2",
        "--early_stopping_patience", "2",
        "--dataloader_num_workers", model_config["dataloader_num_workers"],
        "--seed", "42",
    ]

    print("\n" + "=" * 80)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 80)
    print(f"Modelo:      {model_config['display_name']}")
    print(f"Dataset:     {dataset_config['display_name']}")
    print(f"Experimento: {experiment_name}")
    print(f"Resultados:  {results_dir}")
    print(f"Modelo:      {output_dir}")

    confirm = input("\n¿Lanzar entrenamiento? [s/n]: ").strip().lower()

    if confirm != "s":
        print("Entrenamiento cancelado.")
        return

    print("\n" + "=" * 80)
    print("EJECUTANDO ENTRENAMIENTO")
    print("=" * 80)
    print(" ".join(cmd))

    subprocess.run(cmd, check=True, cwd=ROOT)

    print("\n" + "=" * 80)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 80)
    print(f"Resultados guardados en: {results_dir}")


def main() -> None:
    model_option = show_menu("MODELOS DISPONIBLES", MODELS)
    dataset_option = show_menu("DATASETS DISPONIBLES", DATASETS)

    model_config = MODELS[model_option]
    dataset_config = DATASETS[dataset_option]

    run_training(model_config, dataset_config)


if __name__ == "__main__":
    main()