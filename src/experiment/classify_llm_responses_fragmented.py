#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrada compatible para clasificar respuestas LLM con analisis por fragmentos.

La implementacion vive en:
    experiment_modules/fragmentation.py
    experiment_modules/fragmented_classifier.py
    experiment_modules/fragmented_classifier_menu.py
"""

from __future__ import annotations

from experiment_modules.fragmented_classifier import (
    FragmentedClassifierSettings,
    aggregate_fragment_predictions,
    build_fragments_dataframe,
    classify_fragments,
    classify_full_responses,
    prepare_work_dataframe,
    run_fragmented_classification,
)
from experiment_modules.fragmented_classifier_menu import main_menu


if __name__ == "__main__":
    # Punto de entrada para el modo interactivo de clasificacion fragmentada.
    main_menu()
