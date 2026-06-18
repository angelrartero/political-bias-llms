#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrada para evaluar respuestas de un LLM usando otro LLM como juez.

La implementacion vive en:
    experiment_modules/llm_judge_classifier.py
    experiment_modules/llm_judge_classifier_menu.py
"""

from __future__ import annotations

from experiment_modules.llm_judge_classifier import (
    LlmJudgeClassifierSettings,
    classify_dataframe_with_llm_judge,
    run_llm_judge_classification,
)
from experiment_modules.llm_judge_classifier_menu import main_menu


if __name__ == "__main__":
    main_menu()
