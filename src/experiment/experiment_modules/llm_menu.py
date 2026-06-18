# Define el menu interactivo para lanzar experimentos de prompts contra LLMs.
from __future__ import annotations

import traceback
from typing import Dict, List

from dotenv import load_dotenv

from experiment_modules.llm_config import (
    DEFAULT_MODEL_REGISTRY,
    DEFAULT_SYSTEM_PROMPT,
    ExperimentSettings,
    ModelConfig,
)
from experiment_modules.llm_io import read_prompts
from experiment_modules.llm_providers import PROVIDER_CALLS
from experiment_modules.llm_runner import run_experiment
from experiment_modules.paths import PROJECT_ROOT


def print_settings(settings: ExperimentSettings) -> None:
    """Muestra los parametros actuales del experimento LLM."""
    print("\nCONFIGURACION ACTUAL")
    print("-" * 60)
    print(f"CSV entrada:       {settings.input_csv}")
    print(f"CSV salida base:   {settings.output_csv}")
    print(f"Runs por prompt:   {settings.runs_per_prompt}")
    print(f"Temperature:       {settings.temperature}")
    print(f"Max tokens:        {settings.max_tokens}")
    print(f"Delay segundos:    {settings.delay_seconds}")
    print(f"Max retries:       {settings.max_retries}")
    print(f"Limite prompts:    {settings.limit_prompts}")
    print(f"Resume:            {settings.resume}")
    print(f"Prompt IDs:        {settings.prompt_ids}")
    print(f"System prompt:     {settings.system_prompt}")
    print("-" * 60)


def list_models(registry: Dict[str, ModelConfig]) -> None:
    """Lista los modelos disponibles en el registro local."""
    print("\nMODELOS DISPONIBLES")
    print("-" * 90)
    for key, cfg in registry.items():
        print(f"{key}. {cfg.alias:20s} | provider={cfg.provider:24s} | model={cfg.model}")
    print("-" * 90)


def select_models(registry: Dict[str, ModelConfig]) -> List[ModelConfig]:
    """Permite seleccionar uno, varios o todos los modelos configurados."""
    list_models(registry)
    raw = input("Selecciona modelos por numero separados por coma, o 'all': ").strip()

    if raw.lower() == "all":
        return list(registry.values())

    selected = []
    for item in raw.split(","):
        # Ignora opciones mal escritas sin cancelar toda la seleccion.
        key = item.strip()
        if key in registry:
            selected.append(registry[key])
        else:
            print(f"[WARN] Opcion ignorada: {key}")

    if not selected:
        print("[WARN] No has seleccionado ningun modelo. Se usara el primero.")
        selected = [registry[sorted(registry.keys())[0]]]

    return selected


def add_custom_model(registry: Dict[str, ModelConfig]) -> None:
    """Anade temporalmente un modelo al registro durante la sesion del menu."""
    print("\nAnadir modelo personalizado")
    print("Proveedores soportados:")
    for provider in PROVIDER_CALLS:
        print(f"  - {provider}")

    alias = input("Alias interno, ej. gpt_exp: ").strip()
    provider = input("Provider: ").strip()
    model = input("Nombre exacto del modelo en la API: ").strip()
    api_key_env = input("Variable de entorno de la API key, ej. OPENAI_API_KEY: ").strip()
    base_url = input("Base URL si aplica, si no deja vacio: ").strip() or None

    if not alias or not provider or not model or not api_key_env:
        print("[ERROR] alias, provider, model y api_key_env son obligatorios.")
        return
    if provider not in PROVIDER_CALLS:
        print(f"[ERROR] Provider no soportado: {provider}")
        return

    next_key = str(max([int(k) for k in registry.keys()] + [0]) + 1)
    # El modelo personalizado vive solo en memoria; no modifica llm_config.py.
    registry[next_key] = ModelConfig(alias, provider, model, api_key_env, base_url)
    print(f"Modelo anadido como opcion {next_key}.")


def configure_paths(settings: ExperimentSettings) -> None:
    """Actualiza rutas de CSV de prompts y salida base."""
    input_csv = input(f"CSV entrada [{settings.input_csv}]: ").strip()
    output_csv = input(f"CSV salida base [{settings.output_csv}]: ").strip()

    if input_csv:
        settings.input_csv = input_csv
    if output_csv:
        settings.output_csv = output_csv


def configure_generation(settings: ExperimentSettings) -> None:
    """Configura repeticiones, temperatura, tokens, pausas y reintentos."""
    def ask_int(label: str, current: int) -> int:
        """Lee un entero manteniendo el valor actual si se deja vacio."""
        raw = input(f"{label} [{current}]: ").strip()
        return current if not raw else int(raw)

    def ask_float(label: str, current: float) -> float:
        """Lee un float manteniendo el valor actual si se deja vacio."""
        raw = input(f"{label} [{current}]: ").strip()
        return current if not raw else float(raw)

    settings.runs_per_prompt = ask_int("Runs por prompt", settings.runs_per_prompt)
    settings.temperature = ask_float("Temperature", settings.temperature)
    settings.max_tokens = ask_int("Max tokens de salida", settings.max_tokens)
    settings.delay_seconds = ask_float("Delay entre llamadas", settings.delay_seconds)
    settings.max_retries = ask_int("Max retries por llamada", settings.max_retries)

    raw_limit = input(f"Limite de prompts para prueba, vacio = todos [{settings.limit_prompts}]: ").strip()
    if raw_limit == "":
        settings.limit_prompts = settings.limit_prompts
    elif raw_limit.lower() in {"none", "todos", "all"}:
        settings.limit_prompts = None
    else:
        settings.limit_prompts = int(raw_limit)

    resume = input(f"Reanudar sin repetir success s/n [{'s' if settings.resume else 'n'}]: ").strip()
    if resume.lower() in {"s", "si", "y", "yes"}:
        settings.resume = True
    elif resume.lower() in {"n", "no"}:
        settings.resume = False

    prompt_ids = input(f"Lista prompt_id separada por comas, vacio=todos [{settings.prompt_ids}]: ").strip()
    if prompt_ids:
        if prompt_ids.lower() in {"none", "todos", "all"}:
            settings.prompt_ids = None
        else:
            settings.prompt_ids = prompt_ids


def configure_system_prompt(settings: ExperimentSettings) -> None:
    """Permite mantener, restaurar, vaciar o reescribir el system prompt."""
    print("\nSystem prompt actual:")
    print(settings.system_prompt)
    print("\nOpciones:")
    print("1. Mantener")
    print("2. Usar system prompt por defecto")
    print("3. Dejar vacio")
    print("4. Escribir uno nuevo")
    opt = input("Elige opcion: ").strip()

    if opt == "2":
        settings.system_prompt = DEFAULT_SYSTEM_PROMPT
    elif opt == "3":
        settings.system_prompt = ""
    elif opt == "4":
        print("Escribe el nuevo system prompt. Termina con una linea vacia:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        settings.system_prompt = "\n".join(lines).strip()


def preview_prompts(settings: ExperimentSettings) -> None:
    """Muestra las primeras filas del CSV de prompts y sus recuentos."""
    try:
        df = read_prompts(settings.input_csv, settings.delimiter)
    except Exception as exc:
        print(f"[ERROR] No se pudo leer el CSV: {exc}")
        return

    print("\nPrimeras filas del CSV:")
    cols = ["prompt_id", "topic", "prompt_type", "induced_frame", "axis", "prompt"]
    existing = [c for c in cols if c in df.columns]
    print(df[existing].head(10).to_string(index=False))
    print(f"\nTotal prompts: {len(df)}")
    print("\nRecuento por topic:")
    print(df["topic"].value_counts().sort_index().to_string())
    print("\nRecuento por prompt_type:")
    print(df["prompt_type"].value_counts().sort_index().to_string())


def main_menu() -> None:
    """Bucle interactivo para configurar y lanzar experimentos LLM."""
    load_dotenv(PROJECT_ROOT / ".env")

    settings = ExperimentSettings()
    registry = dict(DEFAULT_MODEL_REGISTRY)
    # Arranca con un modelo por defecto para que el menu sea ejecutable de inmediato.
    selected_models: List[ModelConfig] = [registry["1"]]

    while True:
        print("\n" + "=" * 72)
        print(" MENU EXPERIMENTO LLM - PROMPTS POLITICOS")
        print("=" * 72)
        print("1. Ver configuracion actual")
        print("2. Configurar rutas CSV entrada/salida base")
        print("3. Previsualizar CSV de prompts")
        print("4. Seleccionar modelos")
        print("5. Anadir modelo personalizado")
        print("6. Configurar generacion")
        print("7. Configurar system prompt")
        print("8. Dry run")
        print("9. Ejecutar experimento")
        print("0. Salir")
        print("-" * 72)
        print("Modelos seleccionados:", ", ".join(m.alias for m in selected_models))

        opt = input("Elige opcion: ").strip()

        try:
            if opt == "1":
                print_settings(settings)
                list_models(registry)
            elif opt == "2":
                configure_paths(settings)
            elif opt == "3":
                preview_prompts(settings)
            elif opt == "4":
                selected_models = select_models(registry)
            elif opt == "5":
                add_custom_model(registry)
            elif opt == "6":
                configure_generation(settings)
            elif opt == "7":
                configure_system_prompt(settings)
            elif opt == "8":
                run_experiment(settings, selected_models, dry_run=True)
            elif opt == "9":
                if input("Ejecutar experimento completo? (s/N): ").strip().lower() == "s":
                    run_experiment(settings, selected_models, dry_run=False)
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
