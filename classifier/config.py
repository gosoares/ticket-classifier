"""Configurações centralizadas do classificador."""

import os

from dotenv import load_dotenv

load_dotenv()

# RAG
K_SIMILAR = 5  # Tickets similares a recuperar
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Prompt
MIN_SAMPLES_PER_CLASS = 1  # Mínimo de tickets por classe no prompt

# Avaliação
VALIDATION_SIZE = 200  # Tickets de validação (25 por classe × 8 classes)
RANDOM_STATE = 123  # Seed para reprodutibilidade

# LLM Reasoning (optional)
REASONING_EFFORT: str | None = os.environ.get("LLM_REASONING_EFFORT") or None
