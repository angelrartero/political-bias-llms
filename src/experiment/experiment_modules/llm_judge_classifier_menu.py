# Menu interactivo para evaluar respuestas de un LLM con otro LLM como juez.
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from experiment_modules.classification_common import read_csv_input
from experiment_modules.llm_config import DEFAULT_MODEL_REGISTRY, ModelConfig
from experiment_modules.llm_menu import add_custom_model, list_models
from experiment_modules.llm_judge_classifier import (
    LlmJudgeClassifierSettings,
    build_llm_judge_output_paths,
    get_registry_model,
    run_llm_judge_classification,
)
from experiment_modules.paths import PROJECT_ROOT, RESPONSES_DIR
from experiment_modules.response_classifier_menu import (
    ask_number_option,
    list_model_response_dirs,
    list_response_csvs,
)


def print_settings(settings: LlmJudgeClassifierSettings, registry: Dict[str, ModelConfig]) -> None:
    """Muestra la configuracion activa de evaluacion cruzada."""
    try:
        judge_config = get_registry_model(settings.judge_model_key, registry)
    except Exception:
        judge_config = None

    print("\nCONFIGURACION ACTUAL")
    print("-" * 70)
    print(f"CSV respuestas:       {settings.input_csv}")
    print(f"CSV salida:           {settings.output_csv or '(auto)'}")
    print(f"JSON resumen:         {settings.summary_json or '(auto)'}")
    if judge_config:
        print(f"Modelo juez:          {judge_config.alias} | {judge_config.provider} | {judge_config.model}")
    else:
        print(f"Modelo juez:          {settings.judge_model_key}")
    print(f"Columna texto:        {settings.text_column}")
    print(f"Success only:         {settings.success_only}")
    print(f"Temperature juez:     {settings.temperature}")
    print(f"Max tokens juez:      {settings.max_tokens}")
    print(f"Delay segundos:       {settings.delay_seconds}")
    print(f"Max retries:          {settings.max_retries}")
    print(f"Limite filas:         {settings.limit_rows}")
    print("-" * 70)
    list_models(registry)


def configure_paths(settings: LlmJudgeClassifierSettings) -> None:
    """Permite cambiar entrada y salidas manualmente."""
    input_csv = input(f"CSV respuestas [{settings.input_csv}]: ").strip()
    output_csv = input(f"CSV salida [{settings.output_csv or 'auto'}]: ").strip()
    summary_json = input(f"JSON resumen [{settings.summary_json or 'auto'}]: ").strip()

    if input_csv:
        settings.input_csv = input_csv
        if not output_csv:
            settings.output_csv = ""
            settings.summary_json = ""
    if output_csv:
        settings.output_csv = output_csv
        if not summary_json:
            settings.summary_json = str(Path(output_csv).with_suffix(".summary.json"))
    if summary_json:
        settings.summary_json = summary_json


def configure_input_from_model_output(
    settings: LlmJudgeClassifierSettings,
    registry: Dict[str, ModelConfig],
) -> None:
    """Permite escoger un CSV de respuestas desde data/responses."""
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

    selected_dir_idx = ask_number_option(len(model_dirs), "Elige carpeta de modelo evaluado")
    if selected_dir_idx is None:
        return

    model_dir = model_dirs[selected_dir_idx - 1]
    csvs = list_response_csvs(model_dir)

    if not csvs:
        print(f"\nNo hay CSV de respuestas en: {model_dir}")
        return

    print(f"\nCSV disponibles en {model_dir.name}")
    print("-" * 70)
    for idx, csv_path in enumerate(csvs, start=1):
        print(f"{idx}. {csv_path.name}")
    print("-" * 70)

    selected_csv_idx = ask_number_option(len(csvs), "Elige CSV de respuestas")
    if selected_csv_idx is None:
        return

    settings.input_csv = str(csvs[selected_csv_idx - 1])
    settings.output_csv = ""
    settings.summary_json = ""
    build_llm_judge_output_paths(settings, get_registry_model(settings.judge_model_key, registry))

    print("\nSeleccion aplicada:")
    print(f"CSV respuestas: {settings.input_csv}")
    print(f"CSV salida:     {settings.output_csv}")
    print(f"JSON resumen:   {settings.summary_json}")


def select_judge_model(settings: LlmJudgeClassifierSettings, registry: Dict[str, ModelConfig]) -> None:
    """Selecciona el LLM que actuara como juez."""
    list_models(registry)
    raw = input("Selecciona modelo juez por numero o alias: ").strip()
    if not raw:
        return

    if raw in registry:
        settings.judge_model_key = raw
    else:
        aliases = {cfg.alias: key for key, cfg in registry.items()}
        if raw not in aliases:
            print(f"[ERROR] Modelo no encontrado: {raw}")
            return
        settings.judge_model_key = aliases[raw]

    settings.output_csv = ""
    settings.summary_json = ""
    build_llm_judge_output_paths(settings, get_registry_model(settings.judge_model_key, registry))


def configure_judge_inference(settings: LlmJudgeClassifierSettings) -> None:
    """Configura parametros de llamada al juez."""
    temperature = input(f"Temperature juez [{settings.temperature}]: ").strip()
    max_tokens = input(f"Max tokens juez [{settings.max_tokens}]: ").strip()
    delay_seconds = input(f"Delay entre llamadas [{settings.delay_seconds}]: ").strip()
    max_retries = input(f"Max retries [{settings.max_retries}]: ").strip()
    limit_rows = input(f"Limite filas, vacio=todas [{settings.limit_rows}]: ").strip()
    success_only = input(f"Filtrar solo status=success s/n [{'s' if settings.success_only else 'n'}]: ").strip()

    if temperature:
        settings.temperature = float(temperature)
    if max_tokens:
        settings.max_tokens = int(max_tokens)
    if delay_seconds:
        settings.delay_seconds = float(delay_seconds)
    if max_retries:
        settings.max_retries = int(max_retries)
    if limit_rows:
        if limit_rows.lower() in {"none", "todos", "all"}:
            settings.limit_rows = None
        else:
            settings.limit_rows = int(limit_rows)
    if success_only.lower() in {"s", "si", "y", "yes"}:
        settings.success_only = True
    elif success_only.lower() in {"n", "no"}:
        settings.success_only = False


def preview_input(settings: LlmJudgeClassifierSettings) -> None:
    """Muestra vista rapida del CSV que se va a evaluar."""
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


def main_menu() -> None:
    """Bucle interactivo para evaluacion cruzada LLM -> LLM juez."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings = LlmJudgeClassifierSettings()
    registry = dict(DEFAULT_MODEL_REGISTRY)

    while True:
        print("\n" + "=" * 72)
        print(" MENU EVALUACION CRUZADA: LLM JUEZ SOBRE RESPUESTAS LLM")
        print("=" * 72)
        print("1. Ver configuracion actual")
        print("2. Configurar rutas CSV")
        print("3. Elegir respuestas por modelo desde data/responses")
        print("4. Seleccionar modelo juez")
        print("5. Anadir modelo juez personalizado")
        print("6. Previsualizar CSV de respuestas")
        print("7. Configurar inferencia del juez")
        print("8. Ejecutar evaluacion cruzada")
        print("0. Salir")
        print("-" * 72)

        opt = input("Elige opcion: ").strip()

        try:
            if opt == "1":
                print_settings(settings, registry)
            elif opt == "2":
                configure_paths(settings)
            elif opt == "3":
                configure_input_from_model_output(settings, registry)
            elif opt == "4":
                select_judge_model(settings, registry)
            elif opt == "5":
                add_custom_model(registry)
            elif opt == "6":
                preview_input(settings)
            elif opt == "7":
                configure_judge_inference(settings)
            elif opt == "8":
                if input("Ejecutar evaluacion cruzada? (s/N): ").strip().lower() == "s":
                    run_llm_judge_classification(settings, registry)
                else:
                    print("Cancelado.")
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
