#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrada compatible para ejecutar experimentos de prompts politicos contra APIs LLM.

La implementacion vive en:
    experiment_modules/llm_config.py
    experiment_modules/llm_io.py
    experiment_modules/llm_providers.py
    experiment_modules/llm_runner.py
    experiment_modules/llm_menu.py
"""

from __future__ import annotations

from experiment_modules.llm_config import (
    DEFAULT_MODEL_REGISTRY,
    DEFAULT_SYSTEM_PROMPT,
    ExperimentSettings,
    ModelConfig,
)
from experiment_modules.llm_menu import main_menu
from experiment_modules.llm_runner import call_with_retries, run_experiment


if __name__ == "__main__":
    # Punto de entrada para configurar y ejecutar llamadas a modelos LLM.
    main_menu()
