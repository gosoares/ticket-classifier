"""Configurações centralizadas do classificador."""

# RAG
K_SIMILAR = 5  # Tickets similares a recuperar
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Prompt
MIN_SAMPLES_PER_CLASS = 1  # Mínimo de tickets por classe no prompt

# Avaliação
TEST_SIZE = 200  # Tickets de teste
RANDOM_STATE = 123  # Seed para reprodutibilidade

# LLM
LLM_MODEL = "google/gemini-2.0-flash-exp:free"
