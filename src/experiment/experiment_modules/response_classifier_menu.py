# Define el menu interactivo para clasificar respuestas completas de LLM.
from __future__ import annotations

import traceback
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from experiment_modules.classification_common import read_csv_input
from experiment_modules.classification_paths import build_classified_output_csv, build_summary_json
from experiment_modules.paths import PROJECT_ROOT, RESPONSES_DIR
from experiment_modules.response_classifier import ClassifierSettings, run_classification


def print_settings(settings: ClassifierSettings) -> None:
    """Muestra la configuracion activa del clasificador simple."""
    print("\nCONFIGURACION ACTUAL")
    print("-" * 70)
    print(f"CSV entrada:       {settings.input_csv}")
    print(f"CSV salida:        {settings.output_csv}")
    print(f"JSON resumen:      {settings.summary_json}")
    print(f"Modelo:            {settings.model_path}")
    print(f"Columna texto:     {settings.text_column}")
    print(f"Success only:      {settings.success_only}")
    print(f"Batch size:        {settings.batch_size}")
    print(f"Max length:        {settings.max_length}")
    print(f"Device:            {settings.device}")
    print("-" * 70)


def configure_paths(settings: ClassifierSettings) -> None:
    """Permite cambiar entrada, salida, resumen y ruta del modelo."""
    input_csv = input(f"CSV entrada [{settings.input_csv}]: ").strip()
    output_csv = input(f"CSV salida [{settings.output_csv}]: ").strip()
    summary_json = input(f"JSON resumen [{settings.summary_json}]: ").strip()
    model_path = input(f"Ruta modelo [{settings.model_path}]: ").strip()

    if input_csv:
        settings.input_csv = input_csv

        # Si no se indica salida, se calcula una ruta coherente con la entrada.
        if not output_csv:
            settings.output_csv = str(build_classified_output_csv(settings.input_csv))
            settings.summary_json = str(build_summary_json(settings.output_csv))

    if output_csv:
        settings.output_csv = output_csv
        if not summary_json:
            settings.summary_json = str(build_summary_json(settings.output_csv))
    if summary_json:
        settings.summary_json = summary_json
    if model_path:
        settings.model_path = model_path


def list_model_response_dirs() -> list[Path]:
    """Lista carpetas de respuestas generadas por modelo."""
    if not RESPONSES_DIR.exists():
        return []

    return sorted(
        [path for path in RESPONSES_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )


def list_response_csvs(model_dir: Path) -> list[Path]:
    """Devuelve CSV clasificables, excluyendo salidas ya procesadas."""
    csvs = sorted(model_dir.glob("*.csv"), key=lambda path: path.name.lower())
    return [path for path in csvs if is_classifiable_response_csv(path)]


def is_classifiable_response_csv(path: Path) -> bool:
    """Indica si un CSV contiene respuestas reales de LLM clasificables."""
    name = path.name.lower()
    excluded_prefixes = ("classified_", "fragmented_classified_", "fragments_", "llm_judge_")

    if name.endswith(".summary.json"):
        return False
    if any(name.startswith(prefix) for prefix in excluded_prefixes):
        return False
    if "dry_run" in name:
        return False

    return path.suffix.lower() == ".csv"


def list_all_response_csvs() -> list[Path]:
    """Lista todos los CSV de respuestas reales, sin incluir dry-runs."""
    return [
        csv_path
        for model_dir in list_model_response_dirs()
        for csv_path in list_response_csvs(model_dir)
    ]


def ask_number_option(max_value: int, prompt: str = "Elige opcion") -> int | None:
    """Lee y valida una opcion numerica del menu."""
    raw = input(f"{prompt}: ").strip()
    if raw == "":
        return None

    try:
        value = int(raw)
    except ValueError:
        print("[ERROR] Debes introducir un numero.")
        return None

    if value < 1 or value > max_value:
        print("[ERROR] Opcion fuera de rango.")
        return None

    return value


def configure_input_from_model_output(settings: ClassifierSettings) -> None:
    """Permite escoger un CSV existente desde data/responses."""
    model_dirs = list_model_response_dirs()

    if not model_dirs:
        print(f"\nNo hay carpetas de modelos en: {RESPONSES_DIR}")
        return

    print("\nRESPUESTAS DISPONIBLES POR MODELO")
    print(f"Raiz: {RESPONSES_DIR}")
    print("-" * 70)
    for idx, model_dir in enumerate(model_dirs, start=1):
        csv_count = len(list_response_csvs(model_dir))
        print(f"{idx}. {model_dir.name} ({csv_count} CSV)")
    print("-" * 70)

    selected_dir_idx = ask_number_option(len(model_dirs), "Elige carpeta de modelo")
    if selected_dir_idx is None:
        return

    model_dir = model_dirs[selected_dir_idx - 1]
    csvs = list_response_csvs(model_dir)

    if not csvs:
        print(f"\nNo hay CSV de respuestas sin clasificar en: {model_dir}")
        return

    print(f"\nCSV disponibles en {model_dir.name}")
    print("-" * 70)
    for idx, csv_path in enumerate(csvs, start=1):
        print(f"{idx}. {csv_path.name}")
    print("-" * 70)

    selected_csv_idx = ask_number_option(len(csvs), "Elige CSV de respuestas")
    if selected_csv_idx is None:
        return

    input_csv = csvs[selected_csv_idx - 1]
    output_csv = build_classified_output_csv(input_csv)

    settings.input_csv = str(input_csv)
    settings.output_csv = str(output_csv)
    settings.summary_json = str(build_summary_json(output_csv))

    print("\nSeleccion aplicada:")
    print(f"CSV entrada:  {settings.input_csv}")
    print(f"CSV salida:   {settings.output_csv}")
    print(f"JSON resumen: {settings.summary_json}")


def configure_default_from_available_model_output(settings: ClassifierSettings) -> None:
    """Selecciona automaticamente el primer CSV disponible si falta el default."""
    if Path(settings.input_csv).exists():
        return

    for model_dir in list_model_response_dirs():
        csvs = list_response_csvs(model_dir)
        if not csvs:
            continue

        input_csv = csvs[0]
        output_csv = build_classified_output_csv(input_csv)

        settings.input_csv = str(input_csv)
        settings.output_csv = str(output_csv)
        settings.summary_json = str(build_summary_json(output_csv))
        return


def build_settings_for_input_csv(base_settings: ClassifierSettings, input_csv: Path) -> ClassifierSettings:
    """Crea una configuracion de clasificacion para un CSV concreto."""
    output_csv = build_classified_output_csv(input_csv)
    return replace(
        base_settings,
        input_csv=str(input_csv),
        output_csv=str(output_csv),
        summary_json=str(build_summary_json(output_csv)),
    )


def run_all_available_classifications(settings: ClassifierSettings) -> None:
    """Clasifica todos los CSV de respuestas reales disponibles en data/responses."""
    response_csvs = list_all_response_csvs()

    if not response_csvs:
        print(f"\nNo hay CSV de respuestas reales en: {RESPONSES_DIR}")
        return

    model_path = Path(settings.model_path)
    if not model_path.exists():
        print("\n[ERROR] No se puede ejecutar el lote porque no existe la carpeta del modelo:")
        print(f"  {model_path}")
        print("Restaura o copia el modelo entrenado antes de lanzar la clasificacion.")
        return

    if not (model_path / "config.json").exists():
        print("\n[ERROR] La carpeta del modelo existe, pero no contiene config.json:")
        print(f"  {model_path}")
        print("Apunta a la carpeta Hugging Face exacta, normalmente best_model/.")
        return

    print("\nCSV que se van a clasificar (dry-run excluidos):")
    print("-" * 70)
    for idx, csv_path in enumerate(response_csvs, start=1):
        print(f"{idx}. {csv_path.relative_to(RESPONSES_DIR)}")
    print("-" * 70)
    print(f"Total: {len(response_csvs)} CSV")

    if input("Ejecutar clasificacion para todos? (s/N): ").strip().lower() != "s":
        print("Cancelado.")
        return

    failures: list[tuple[Path, Exception]] = []

    for idx, input_csv in enumerate(response_csvs, start=1):
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(response_csvs)}] Clasificando: {input_csv.relative_to(RESPONSES_DIR)}")
        print("=" * 70)
        try:
            run_classification(build_settings_for_input_csv(settings, input_csv))
        except Exception as exc:
            failures.append((input_csv, exc))
            print(f"[ERROR] No se pudo clasificar {input_csv}: {type(exc).__name__}: {exc}")

    print("\nEjecucion por lotes finalizada.")
    print(f"CSV procesados correctamente: {len(response_csvs) - len(failures)} / {len(response_csvs)}")
    if failures:
        print("\nFallos:")
        for input_csv, exc in failures:
            print(f"- {input_csv}: {type(exc).__name__}: {exc}")


def configure_inference(settings: ClassifierSettings) -> None:
    """Configura batch size, longitud maxima, dispositivo y filtro success."""
    batch_size = input(f"Batch size [{settings.batch_size}]: ").strip()
    max_length = input(f"Max length [{settings.max_length}]: ").strip()
    device = input(f"Device auto/cuda/cpu [{settings.device}]: ").strip()
    success_only = input(f"Filtrar solo status=success s/n [{'s' if settings.success_only else 'n'}]: ").strip()

    if batch_size:
        settings.batch_size = int(batch_size)
    if max_length:
        settings.max_length = int(max_length)
    if device:
        settings.device = device
    if success_only.lower() in {"s", "si", "y", "yes"}:
        settings.success_only = True
    elif success_only.lower() in {"n", "no"}:
        settings.success_only = False


def preview_input(settings: ClassifierSettings) -> None:
    """Muestra una vista rapida del CSV antes de clasificarlo."""
    try:
        df = read_csv_input(settings.input_csv, settings.delimiter)
    except Exception as exc:
        print(f"[ERROR] No se pudo leer el CSV: {exc}")
        return

    columns_to_show = [
        "response_id",
        "prompt_id",
        "topic",
        "prompt_type",
        "induced_frame",
        "model_alias",
        "model",
        "status",
        "response",
    ]
    existing = [c for c in columns_to_show if c in df.columns]

    print("\nPrimeras filas:")
    print(df[existing].head(5).to_string(index=False))
    print(f"\nTotal filas: {len(df)}")

    if "status" in df.columns:
        print("\nRecuento por status:")
        print(df["status"].value_counts().to_string())
    if "topic" in df.columns:
        print("\nRecuento por topic:")
        print(df["topic"].value_counts().sort_index().to_string())


def main_menu() -> None:
    """Bucle interactivo para clasificar respuestas completas."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings = ClassifierSettings()
    configure_default_from_available_model_output(settings)

    while True:
        print("\n" + "=" * 72)
        print(" MENU CLASIFICACION IDEOLOGICA DE RESPUESTAS LLM")
        print("=" * 72)
        print("1. Ver configuracion actual")
        print("2. Configurar rutas CSV/modelo")
        print("3. Elegir respuestas por modelo desde data/responses")
        print("4. Previsualizar CSV de entrada")
        print("5. Configurar inferencia")
        print("6. Ejecutar clasificacion")
        print("7. Ejecutar clasificacion de todas las respuestas no dry-run")
        print("0. Salir")
        print("-" * 72)

        opt = input("Elige opcion: ").strip()

        try:
            if opt == "1":
                print_settings(settings)
            elif opt == "2":
                configure_paths(settings)
            elif opt == "3":
                configure_input_from_model_output(settings)
            elif opt == "4":
                preview_input(settings)
            elif opt == "5":
                configure_inference(settings)
            elif opt == "6":
                if input("Ejecutar clasificacion? (s/N): ").strip().lower() == "s":
                    run_classification(settings)
                else:
                    print("Cancelado.")
            elif opt == "7":
                run_all_available_classifications(settings)
            elif opt == "0":
                print("Saliendo.")
                break
            else:
                print("Opcion no valida.")
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario.")
        except Exception as exc:
            print("\n[ERROR]")
            print(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
