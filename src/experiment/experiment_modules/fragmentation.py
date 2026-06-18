# Separa respuestas LLM en parrafos, frases o ventanas de tokens.
from __future__ import annotations

import re
from typing import List

from transformers import AutoTokenizer


def split_into_paragraphs(text: str) -> List[str]:
    """Divide el texto conservando primero la estructura de parrafos."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # Prioriza saltos dobles como separadores de parrafo; si no existen,
    # acepta saltos simples para respuestas que usan una linea por bloque.
    parts = re.split(r"\n\s*\n+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts

    parts = re.split(r"\n+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def split_into_sentences(text: str) -> List[str]:
    """Divide el texto en frases usando puntuacion basica como separador."""
    text = str(text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences if sentences else [text]


def group_sentences(sentences: List[str], sentences_per_fragment: int) -> List[str]:
    """Agrupa frases consecutivas para formar fragmentos mas estables."""
    if sentences_per_fragment <= 1:
        return sentences

    grouped = []
    for i in range(0, len(sentences), sentences_per_fragment):
        grouped.append(" ".join(sentences[i:i + sentences_per_fragment]).strip())

    return [g for g in grouped if g]


def merge_short_fragments(fragments: List[str], min_chars: int) -> List[str]:
    """Une fragmentos muy cortos al anterior para evitar trozos poco informativos."""
    if not fragments:
        return []
    if min_chars <= 0:
        return fragments

    merged: List[str] = []
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if merged and len(fragment) < min_chars:
            merged[-1] = f"{merged[-1]} {fragment}".strip()
        else:
            merged.append(fragment)

    return merged


def token_count(text: str, tokenizer: AutoTokenizer) -> int:
    """Cuenta tokens con el mismo tokenizador que usara el clasificador."""
    ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
    return len(ids)


def split_long_fragment_by_tokens(
    fragment: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
    stride: int,
) -> List[str]:
    """Parte fragmentos largos en ventanas que caben dentro del limite del modelo."""
    ids = tokenizer(fragment, truncation=False, add_special_tokens=False)["input_ids"]
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    max_content_tokens = max(1, max_tokens - special_tokens)

    if len(ids) <= max_content_tokens:
        return [fragment]

    windows = []
    step = max_content_tokens - stride
    if step <= 0:
        step = max_content_tokens

    # Si stride > 0, las ventanas se solapan para no perder contexto entre cortes.
    start = 0
    while start < len(ids):
        end = start + max_content_tokens
        decoded = tokenizer.decode(
            ids[start:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if decoded:
            windows.append(decoded)
        if end >= len(ids):
            break
        start += step

    return windows


def fragment_text(text: str, tokenizer: AutoTokenizer, settings: object) -> List[str]:
    """Aplica la estrategia de fragmentacion configurada y normaliza los cortes."""
    text = str(text).strip()
    if not text:
        return []

    if settings.fragment_strategy == "paragraph":
        fragments = split_into_paragraphs(text)
    elif settings.fragment_strategy == "sentence":
        sentences = split_into_sentences(text)
        fragments = group_sentences(sentences, settings.sentences_per_fragment)
    elif settings.fragment_strategy == "paragraph_then_sentence":
        # Estrategia por defecto: respeta parrafos si existen; si la respuesta
        # es un unico bloque, cae a division por frases agrupadas.
        fragments = split_into_paragraphs(text)
        if len(fragments) <= 1:
            sentences = split_into_sentences(text)
            fragments = group_sentences(sentences, settings.sentences_per_fragment)
    else:
        raise ValueError(f"Estrategia de fragmentacion no soportada: {settings.fragment_strategy}")

    fragments = merge_short_fragments(fragments, settings.min_fragment_chars)
    final_fragments: List[str] = []

    # Segunda pasada defensiva: ningun fragmento debe superar el maximo de tokens.
    for fragment in fragments:
        final_fragments.extend(
            split_long_fragment_by_tokens(
                fragment=fragment,
                tokenizer=tokenizer,
                max_tokens=settings.fragment_max_tokens,
                stride=settings.fragment_stride,
            )
        )

    return [f.strip() for f in final_fragments if f.strip()]
