# Orquesta la ejecucion de experimentos LLM con reintentos y guardado incremental.
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from experiment_modules.llm_config import DEFAULT_OUTPUT_PATH, ExperimentSettings, ModelConfig
from experiment_modules.llm_io import (
    append_result_row,
    drop_retryable_rows,
    dumps_usage,
    load_completed_keys,
    normalize_usage,
    now_iso,
    read_prompts,
)
from experiment_modules.llm_providers import PROVIDER_CALLS
from experiment_modules.paths import RESPONSES_DIR


def call_with_retries(
    model_config: ModelConfig,
    prompt: str,
    settings: ExperimentSettings,
) -> Tuple[str, Dict[str, Any], float]:
    """Ejecuta una llamada LLM con reintentos exponenciales y mide latencia."""
    if model_config.provider not in PROVIDER_CALLS:
        raise ValueError(f"Proveedor no soportado: {model_config.provider}")

    provider_fn = PROVIDER_CALLS[model_config.provider]
    last_exc: Optional[BaseException] = None
    start = time.perf_counter()

    for attempt in range(1, settings.max_retries + 1):
        try:
            text, usage = provider_fn(
                model_config,
                prompt,
                settings.system_prompt,
                settings.temperature,
                effective_max_tokens(settings, model_config),
            )
            return text, usage, time.perf_counter() - start
        except Exception as exc:
            last_exc = exc
            if attempt >= settings.max_retries:
                break
            # Backoff exponencial simple para errores temporales de red/cuota.
            sleep_for = settings.retry_base_seconds * (2 ** (attempt - 1))
            print(
                f"\n[WARN] Error con {model_config.alias}. "
                f"Reintento {attempt}/{settings.max_retries}. Esperando {sleep_for:.1f}s"
            )
            print(f"       {type(exc).__name__}: {exc}")
            time.sleep(sleep_for)

    latency = time.perf_counter() - start
    assert last_exc is not None
    raise RuntimeError(f"Fallo tras {settings.max_retries} intentos en {latency:.2f}s") from last_exc


def make_response_id(prompt_id: str, model_alias: str, run: int) -> str:
    """Crea un identificador reproducible para cada prompt/modelo/repeticion."""
    return f"{prompt_id}__{model_alias}__run{run}"


def build_model_output_csv(
    settings: ExperimentSettings,
    model_config: ModelConfig,
    dry_run: bool,
) -> str:
    """Calcula el CSV de salida correspondiente a un modelo concreto."""
    output_name = Path(settings.output_csv).name

    if output_name == Path(DEFAULT_OUTPUT_PATH).name:
        # Con la ruta por defecto, cada modelo escribe en su propia carpeta.
        suffix = "dry_run_responses" if dry_run else "responses"
        output_name = f"{model_config.alias}_{suffix}.csv"

    return str(RESPONSES_DIR / model_config.alias / output_name)


def reset_output_files(output_paths: Dict[str, str]) -> None:
    """Limpia salidas previas antes de lanzar una nueva ejecucion."""
    for output_csv in output_paths.values():
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            path.unlink()


def effective_max_tokens(settings: ExperimentSettings, model_config: ModelConfig) -> int:
    """Aplica limite especifico por modelo si existe."""
    return model_config.max_tokens_override or settings.max_tokens


def parse_prompt_id_filter(raw: Optional[str]) -> set[str]:
    """Convierte una lista separada por comas en conjunto de prompt_id."""
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def is_truncated_finish_reason(finish_reason: Any) -> bool:
    """Detecta finales por limite de longitud entre proveedores."""
    text = str(finish_reason or "").lower()
    markers = ("length", "max_tokens", "max_output", "model_length", "token_limit")
    return any(marker in text for marker in markers)


def run_experiment(
    settings: ExperimentSettings,
    selected_models: List[ModelConfig],
    dry_run: bool = False,
) -> None:
    """Genera respuestas para todos los prompts, modelos y repeticiones."""
    prompts_df = read_prompts(settings.input_csv, settings.delimiter)

    if settings.limit_prompts is not None and settings.limit_prompts > 0:
        # limit_prompts permite pruebas controladas sin tocar el CSV original.
        prompts_df = prompts_df.head(settings.limit_prompts).copy()
    prompt_id_filter = parse_prompt_id_filter(settings.prompt_ids)
    if prompt_id_filter:
        prompts_df = prompts_df[prompts_df["prompt_id"].isin(prompt_id_filter)].copy()
    if dry_run:
        prompts_df = prompts_df.head(settings.dry_run_rows).copy()

    output_paths = {
        # Cada modelo tiene un CSV separado para facilitar analisis posterior.
        model_config.alias: build_model_output_csv(settings, model_config, dry_run)
        for model_config in selected_models
    }
    if settings.resume:
        for output_csv in output_paths.values():
            drop_retryable_rows(output_csv, settings.delimiter)
    else:
        reset_output_files(output_paths)

    total_calls = len(prompts_df) * len(selected_models) * settings.runs_per_prompt

    if dry_run:
        print("\n[DRY RUN] Se ejecutara una prueba reducida.")
    print(f"\nPrompts: {len(prompts_df)}")
    print(f"Modelos: {len(selected_models)}")
    print(f"Runs por prompt: {settings.runs_per_prompt}")
    print(f"Llamadas maximas: {total_calls}")
    print(f"Resume: {settings.resume}")
    print("Salidas:")
    for model_config in selected_models:
        print(f"  {model_config.alias}: {output_paths[model_config.alias]}")
    print()

    progress = tqdm(total=total_calls, desc="Generando respuestas", unit="call")

    for _, prompt_row in prompts_df.iterrows():
        prompt_id = str(prompt_row["prompt_id"])
        prompt_text = str(prompt_row["prompt"])

        for model_config in selected_models:
            output_csv = output_paths[model_config.alias]
            completed_keys = load_completed_keys(output_csv, settings.delimiter) if settings.resume else set()

            for run_number in range(1, settings.runs_per_prompt + 1):
                call_key = (prompt_id, model_config.alias, run_number)
                if call_key in completed_keys:
                    progress.update(1)
                    continue
                max_tokens = effective_max_tokens(settings, model_config)
                row_base = {
                    # Metadatos comunes tanto para llamadas correctas como fallidas.
                    "response_id": make_response_id(prompt_id, model_config.alias, run_number),
                    "created_at": now_iso(),
                    "prompt_id": prompt_id,
                    "topic": prompt_row.get("topic", ""),
                    "prompt_type": prompt_row.get("prompt_type", ""),
                    "induced_frame": prompt_row.get("induced_frame", ""),
                    "axis": prompt_row.get("axis", ""),
                    "nationalism_sensitive": prompt_row.get("nationalism_sensitive", ""),
                    "model_alias": model_config.alias,
                    "provider": model_config.provider,
                    "model": model_config.model,
                    "run": run_number,
                    "temperature": settings.temperature,
                    "max_tokens": max_tokens,
                    "system_prompt": settings.system_prompt,
                    "prompt": prompt_text,
                }

                try:
                    text, usage, latency = call_with_retries(model_config, prompt_text, settings)

                    if not str(text).strip():
                        raise RuntimeError("El proveedor devolvio una respuesta vacia.")

                    # Estandariza uso de tokens antes de escribir el CSV.
                    input_tokens, output_tokens, total_tokens = normalize_usage(usage)
                    finish_reason = usage.get("finish_reason", "")
                    status = "truncated" if is_truncated_finish_reason(finish_reason) else "success"
                    result_row = {
                        **row_base,
                        "response": text,
                        "status": status,
                        "error_type": "",
                        "error_message": "Respuesta truncada por limite de longitud." if status == "truncated" else "",
                        "finish_reason": finish_reason,
                        "latency_seconds": round(latency, 4),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "raw_usage_json": dumps_usage(usage),
                    }
                except Exception as exc:
                    # Los errores tambien se registran como filas para conservar trazabilidad.
                    root = exc.__cause__ if exc.__cause__ is not None else exc
                    result_row = {
                        **row_base,
                        "response": "",
                        "status": "error",
                        "error_type": type(root).__name__,
                        "error_message": str(root),
                        "finish_reason": "",
                        "latency_seconds": "",
                        "input_tokens": "",
                        "output_tokens": "",
                        "total_tokens": "",
                        "raw_usage_json": "",
                    }
                    print("\n[ERROR] Llamada fallida:")
                    print(f"  prompt_id: {prompt_id}")
                    print(f"  model: {model_config.alias} ({model_config.model})")
                    print(f"  error: {type(root).__name__}: {root}")

                append_result_row(output_csv, result_row, settings.delimiter)

                if settings.delay_seconds > 0:
                    # Pausa opcional para reducir presion sobre cuotas/rate limits.
                    time.sleep(settings.delay_seconds)
                progress.update(1)

    progress.close()
    print("\nExperimento finalizado.")
