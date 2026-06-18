# Implementa los clientes para llamar a proveedores de APIs LLM.
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Tuple

from experiment_modules.llm_config import ModelConfig


GEMINI_2_5_PRO_THINKING_BUDGET = 128


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """Construye mensajes con formato chat compatible con OpenAI y similares."""
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})
    return messages


def extract_usage_dict(obj: Any) -> Dict[str, Any]:
    """Extrae usage de respuestas SDK aunque vengan como objeto o diccionario."""
    if obj is None:
        return {}

    usage = getattr(obj, "usage", None)
    if usage is None and isinstance(obj, dict):
        usage = obj.get("usage")
    if usage is None:
        return {}

    if hasattr(usage, "model_dump"):
        # SDKs basados en pydantic v2.
        return usage.model_dump()
    if hasattr(usage, "dict"):
        # SDKs basados en pydantic v1.
        return usage.dict()
    if isinstance(usage, dict):
        return usage

    result = {}
    for key in ["input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"]:
        if hasattr(usage, key):
            result[key] = getattr(usage, key)
    return result


def model_supports_sampling_params(model: str) -> bool:
    """Detecta modelos que no aceptan parametros como temperature."""
    model_name = model.lower()
    unsupported_prefixes = (
        "gpt-5",
        "o1",
        "o3",
        "o4",
    )
    return not model_name.startswith(unsupported_prefixes)


def anthropic_model_supports_temperature(model: str) -> bool:
    """Detecta modelos Anthropic que no aceptan temperature."""
    model_name = model.lower()
    return not model_name.startswith("claude-opus-4-8")


def is_unsupported_parameter_error(exc: Exception, parameter: str) -> bool:
    """Reconoce errores de proveedor por parametros no soportados."""
    error_text = str(exc).lower()
    parameter = parameter.lower()
    return (
        "unsupported parameter" in error_text
        and parameter in error_text
    ) or f"{parameter}' is not supported" in error_text or (
        parameter in error_text
        and "deprecated" in error_text
    )


def resolve_base_url(base_url: str | None) -> str | None:
    """Permite declarar base_url como env:NOMBRE_VARIABLE en llm_config.py."""
    if base_url and base_url.startswith("env:"):
        env_name = base_url.removeprefix("env:")
        value = os.environ.get(env_name)
        if not value:
            raise RuntimeError(f"Falta variable de entorno: {env_name}")
        return value
    return base_url


def is_ollama_base_url(base_url: str | None) -> bool:
    """Detecta endpoints Ollama para no exigir una API key real."""
    if not base_url:
        return False
    return "11434" in base_url or "ollama" in base_url.lower()


def extract_gemini_text(response: Any) -> str:
    """Extrae texto de Gemini probando primero .text y luego candidates."""
    text = getattr(response, "text", "") or ""
    if text.strip():
        return text.strip()

    parts_text: List[str] = []
    candidates = getattr(response, "candidates", None) or []

    for candidate in candidates:
        # Algunas respuestas de Gemini solo exponen texto en content.parts.
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None

        for part in parts or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts_text.append(part_text)

    return "\n".join(parts_text).strip()


def call_anthropic_messages(
    config: ModelConfig,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Llama a Anthropic Messages y devuelve texto junto con uso de tokens."""
    from anthropic import Anthropic, BadRequestError

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Falta variable de entorno: {config.api_key_env}")

    kwargs = {
        "model": config.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt.strip()}],
    }
    if anthropic_model_supports_temperature(config.model):
        kwargs["temperature"] = temperature
    if system_prompt.strip():
        kwargs["system"] = system_prompt.strip()

    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(**kwargs)
    except BadRequestError as exc:
        if "temperature" in kwargs and is_unsupported_parameter_error(exc, "temperature"):
            print(
                "\n[WARN] Este modelo Anthropic no acepta temperature. "
                "Reintentando sin temperature..."
            )
            kwargs.pop("temperature", None)
            response = client.messages.create(**kwargs)
        else:
            raise
    text = "\n".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()

    response_usage = getattr(response, "usage", None)
    usage = {}
    if response_usage is not None:
        # Anthropic separa input/output; aqui se calcula total si ambos existen.
        input_tokens = getattr(response_usage, "input_tokens", None)
        output_tokens = getattr(response_usage, "output_tokens", None)
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None,
        }
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason:
        usage["finish_reason"] = stop_reason

    return text, usage


def call_openai_responses(
    config: ModelConfig,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Llama a OpenAI Responses API gestionando modelos sin temperature."""
    from openai import BadRequestError, OpenAI

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Falta variable de entorno: {config.api_key_env}")

    client = OpenAI(api_key=api_key)
    request_kwargs = {
        "model": config.model,
        "input": build_messages(system_prompt, prompt),
        "max_output_tokens": max_tokens,
    }

    if model_supports_sampling_params(config.model):
        request_kwargs["temperature"] = temperature

    try:
        response = client.responses.create(**request_kwargs)
    except BadRequestError as exc:
        # Algunos modelos rechazan temperature aunque el resto de parametros sea valido.
        if "temperature" in request_kwargs and is_unsupported_parameter_error(exc, "temperature"):
            print(
                "\n[WARN] Este modelo OpenAI no acepta temperature. "
                "Reintentando sin temperature..."
            )
            request_kwargs.pop("temperature", None)
            response = client.responses.create(**request_kwargs)
        else:
            raise

    usage = extract_usage_dict(response)
    finish_reason = getattr(response, "status", None)
    incomplete_details = getattr(response, "incomplete_details", None)
    if incomplete_details is not None:
        reason = getattr(incomplete_details, "reason", None)
        if reason:
            finish_reason = reason
    if finish_reason:
        usage["finish_reason"] = finish_reason
    return (getattr(response, "output_text", "") or "").strip(), usage


def call_xai_chat(
    config: ModelConfig,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Llama a xAI Grok mediante Responses API compatible con el SDK de OpenAI."""
    from openai import BadRequestError, OpenAI

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Falta variable de entorno: {config.api_key_env}")

    client = OpenAI(api_key=api_key, base_url=resolve_base_url(config.base_url))
    request_kwargs = {
        "model": config.model,
        "input": build_messages(system_prompt, prompt),
        "max_output_tokens": max_tokens,
        "store": False,
    }

    if model_supports_sampling_params(config.model):
        request_kwargs["temperature"] = temperature

    try:
        response = client.responses.create(**request_kwargs)
    except BadRequestError as exc:
        if "temperature" in request_kwargs and is_unsupported_parameter_error(exc, "temperature"):
            print(
                "\n[WARN] Este modelo xAI no acepta temperature. "
                "Reintentando sin temperature..."
            )
            request_kwargs.pop("temperature", None)
            response = client.responses.create(**request_kwargs)
        else:
            raise

    usage = extract_usage_dict(response)
    finish_reason = getattr(response, "status", None)
    incomplete_details = getattr(response, "incomplete_details", None)
    if incomplete_details is not None:
        reason = getattr(incomplete_details, "reason", None)
        if reason:
            finish_reason = reason
    if finish_reason:
        usage["finish_reason"] = finish_reason
    return (getattr(response, "output_text", "") or "").strip(), usage


def call_openai_compatible_chat(
    config: ModelConfig,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Llama a endpoints tipo OpenAI Chat Completions, incluidos OpenRouter y Ollama."""
    from openai import BadRequestError, OpenAI

    base_url = resolve_base_url(config.base_url)
    api_key = os.environ.get(config.api_key_env)
    if not api_key and is_ollama_base_url(base_url):
        # Ollama local/remoto no suele requerir autenticacion, pero el SDK exige api_key.
        api_key = "ollama"
    if not api_key:
        raise RuntimeError(f"Falta variable de entorno: {config.api_key_env}")

    default_headers = {}
    if base_url and "openrouter.ai" in base_url:
        # OpenRouter recomienda cabeceras de referencia para identificar la app.
        default_headers = {
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "TFG Political Bias Experiment",
        }

    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
    request_kwargs = {
        "model": config.model,
        "messages": build_messages(system_prompt, prompt),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if base_url and "openrouter.ai" in base_url and config.model != "openrouter/free":
        # Evita gastar tokens en razonamiento cuando el modelo permite desactivarlo.
        request_kwargs["extra_body"] = {"reasoning": {"effort": "none"}}

    try:
        response = client.chat.completions.create(**request_kwargs)
    except BadRequestError as exc:
        # Reintentos adaptativos ante diferencias entre proveedores compatibles.
        error_text = str(exc)
        if is_unsupported_parameter_error(exc, "temperature"):
            print(
                "\n[WARN] El endpoint no acepta temperature para este modelo. "
                "Reintentando sin temperature..."
            )
            request_kwargs.pop("temperature", None)
            response = client.chat.completions.create(**request_kwargs)
        elif "Reasoning is mandatory" in error_text or "cannot be disabled" in error_text:
            print("\n[WARN] El endpoint no permite desactivar reasoning. Reintentando sin extra_body...")
            request_kwargs.pop("extra_body", None)
            response = client.chat.completions.create(**request_kwargs)
        else:
            raise

    text = response.choices[0].message.content or ""
    usage = extract_usage_dict(response)
    finish_reason = getattr(response.choices[0], "finish_reason", None)
    if finish_reason:
        usage["finish_reason"] = finish_reason
    return text.strip(), usage


def call_gemini_generate_content(
    config: ModelConfig,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Llama a Gemini Generate Content y adapta su uso de tokens al esquema comun."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Falta variable de entorno: {config.api_key_env}")

    thinking_budget = None
    max_output_tokens = max_tokens

    if config.model.lower().startswith("gemini-2.5-pro"):
        # Se reserva un pequeno presupuesto para thinking y se suma al maximo total.
        thinking_budget = GEMINI_2_5_PRO_THINKING_BUDGET
        max_output_tokens = max_tokens + thinking_budget

    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget,
        )

    gen_config = types.GenerateContentConfig(**config_kwargs)
    if system_prompt.strip():
        gen_config.system_instruction = system_prompt.strip()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=config.model,
        contents=prompt,
        config=gen_config,
    )

    usage = {}
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is not None:
        # Gemini usa nombres propios; se guardan y ademas se normalizan.
        if hasattr(usage_metadata, "model_dump"):
            usage = usage_metadata.model_dump()
        elif hasattr(usage_metadata, "dict"):
            usage = usage_metadata.dict()
        else:
            for key in ["prompt_token_count", "candidates_token_count", "total_token_count"]:
                if hasattr(usage_metadata, key):
                    usage[key] = getattr(usage_metadata, key)

        if "prompt_token_count" in usage:
            usage["input_tokens"] = usage.get("prompt_token_count")
        if "candidates_token_count" in usage:
            usage["output_tokens"] = usage.get("candidates_token_count")
        if "total_token_count" in usage:
            usage["total_tokens"] = usage.get("total_token_count")
        finish_reasons = [
            str(getattr(candidate, "finish_reason", ""))
            for candidate in (getattr(response, "candidates", None) or [])
            if str(getattr(candidate, "finish_reason", "")).strip()
        ]
        if finish_reasons:
            usage["finish_reason"] = "|".join(finish_reasons)

    text = extract_gemini_text(response)

    if not text:
        finish_reasons = [
            str(getattr(candidate, "finish_reason", ""))
            for candidate in (getattr(response, "candidates", None) or [])
        ]
        raise RuntimeError(
            "Gemini no devolvio texto visible. "
            f"finish_reasons={finish_reasons}; usage={usage}"
        )

    return text, usage


def call_mistral_chat(
    config: ModelConfig,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Llama al endpoint chat de Mistral."""
    try:
        from mistralai import Mistral
    except ImportError:
        from mistralai.client import Mistral  # type: ignore

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Falta variable de entorno: {config.api_key_env}")

    response = Mistral(api_key=api_key).chat.complete(
        model=config.model,
        messages=build_messages(system_prompt, prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = extract_usage_dict(response)
    finish_reason = getattr(response.choices[0], "finish_reason", None)
    if finish_reason:
        usage["finish_reason"] = finish_reason
    return (response.choices[0].message.content or "").strip(), usage


def call_provider_pending(
    config: ModelConfig,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Marca proveedores registrados pero aun no implementados."""
    raise NotImplementedError(
        f"Proveedor pendiente de implementar: {config.provider} "
        f"(alias={config.alias}, model={config.model})"
    )


PROVIDER_CALLS: Dict[str, Callable[[ModelConfig, str, str, float, int], Tuple[str, Dict[str, Any]]]] = {
    # Tabla central para que el runner no tenga que conocer detalles de cada API.
    "openai_responses": call_openai_responses,
    "anthropic_messages": call_anthropic_messages,
    "openai_compatible_chat": call_openai_compatible_chat,
    "gemini_generate_content": call_gemini_generate_content,
    "mistral_chat": call_mistral_chat,
    "deepseek_chat": call_provider_pending,
    "xai_chat": call_xai_chat,
}
