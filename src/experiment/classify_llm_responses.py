#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrada compatible para clasificar respuestas LLM completas.

La implementacion vive en:
    experiment_modules/response_classifier.py
    experiment_modules/response_classifier_menu.py
"""

from __future__ import annotations

from experiment_modules.response_classifier import (
    ClassifierSettings,
    classify_dataframe,
    run_classification,
)
from experiment_modules.response_classifier_menu import main_menu


if __name__ == "__main__":
    # Mantiene este archivo como punto de entrada corto y delega el flujo al menu.
    main_menu()
