# IT Service Ticket Classification

Sistema de classificação automática de tickets de suporte de TI com ML (TF-IDF + LinearSVC) e geração de **justificativas**:

- **Linear (padrão):** determinístico, rápido e sem dependência de LLM
- **LLM + RAG (opcional):** justificativas em linguagem natural condicionadas a evidências recuperadas

## Objetivo

- **Entrada:** texto do ticket (string)
- **Saída:** `{"classe": "...", "justificativa": "..."}` (classe vem do ML; justificativa pode ser linear ou via LLM)

O sistema classifica tickets em 8 categorias e fornece uma justificativa explicando o motivo da classificação.

## Pipeline (LangGraph)

```mermaid
graph LR;
  T[Ticket] --> V[vectorize - TF-IDF];
  V --> C[classify - LinearSVC];
  C --> R[retrieve - RAG];
  R --> J[justify - LLM or linear];
  J --> O[output - classe e justificativa];
```

## Notebooks

- `notebooks/analysis.ipynb`: análise exploratória (EDA).
- `notebooks/classificators.ipynb`: testes de classificadores (TF-IDF, embeddings, RAG) com métricas no conjunto de teste.
- `notebooks/main.ipynb`: visão geral, prompts, pipeline e avaliação final (usa o classificador escolhido: **TF-IDF + LinearSVC**).

## Módulos Python

Principais arquivos em `classifier/`:

| Arquivo | Responsabilidade |
|---------|------------------|
| `config.py` | Configurações globais (K_SIMILAR, VALIDATION_SIZE, RANDOM_STATE, etc.) |
| `data.py` | Carregamento do dataset + split treino/teste/validação (balanceada) |
| `features.py` | Extração de features TF-IDF (word n-grams; char n-grams opcional) |
| `classifiers.py` | Candidatos de classificação (TF-IDF, embeddings, RAG-vote, etc.) |
| `rag.py` | Embeddings (sentence-transformers) + retriever (similaridade cosseno) |
| `justifiers.py` | Estratégias de justificativa (linear evidence ou LLM+RAG) |
| `llm.py` | Client OpenAI-compatible + prompts + gerador de conclusões |
| `conclusion.py` | Payload de conclusão + prompts (opcional, LLM-only) |
| `metrics.py` | Métricas de avaliação (accuracy, F1, Cohen's Kappa, MCC, matriz de confusão) |
| `runner.py` | Orquestração do pipeline (train → classify → justify → evaluate → save) |
| `graph.py` | LangGraph: definição do grafo de estados e nós do pipeline |
| `artifacts.py` | Persistência de modelos, metadata e índice RAG |
| `schemas.py` | Esquemas compartilhados (JustificationResult, TokenUsage, etc.) |
| `logging_config.py` | Configuração de logging (terminal + arquivo) |

## Requisitos

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Provider LLM compatível com OpenAI (**opcional**, apenas se usar justificativa com LLM)

## Instalação

```bash
git clone <repo>
cd ticket-classifier
uv sync
```

## Configuração (somente se usar LLM)

### 1. Criar arquivo `.env`

```bash
cp .env.example .env
```

### 2. Configurar variáveis de ambiente

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `LLM_BASE_URL` | URL da API | `http://localhost:11434/v1` |
| `LLM_MODEL` | Nome do modelo | `gemma2:2b` |
| `LLM_API_KEY` | Chave da API | `ollama` |
| `LLM_REASONING_EFFORT` | Nível de reasoning (padrão: desativado) | `low`, `medium`, `high`, ou vazio para desativar |
| `JUSTIFICATION` | Modo de justificativa padrão (`linear`/`llm`) | `linear` |

**Critérios do modelo escolhido (padrão):**
- Gratuito e local (Ollama)
- Não exige hardware avançado (modelo pequeno)
- Baixa latência e custo previsível (sem dependência de provedores externos)

### Exemplos por Provider

**OpenRouter:**
```env
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-...
LLM_MODEL=openai/gpt-4o-mini
```

**Ollama (local):**
```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=gemma2:2b
```

**OpenAI:**
```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

## Uso

O CLI possui **dois comandos principais**:

- `train`: treina o modelo, calcula métricas na validação e salva os artefatos em `artifacts/` para reutilização.
- `eval`: carrega os artefatos salvos e retorna a classificação e justificativa de **um** ticket.

### Via CLI

```bash
# Treinar modelo e avaliar classificação (sem justificativas por padrão)
uv run train dataset.csv

# Ver todas as opções
uv run train --help

# Treinar com avaliação e justificativas lineares
uv run train dataset.csv --with-justifications --justification linear

# Treinar com LLM (salva índice RAG automaticamente)
uv run train dataset.csv --with-justifications --justification llm

# Avaliar um ticket (saída JSON do desafio)
uv run eval "Meu laptop não liga"

# Avaliar lendo do stdin
cat ticket.txt | uv run eval

# Avaliar com LLM (requer índice RAG salvo e .env configurado)
uv run eval --llm "Preciso de acesso ao sistema"
```

**Opções CLI (train):**

| Opção | Descrição | Default |
|-------|-----------|---------|
| `dataset` | Caminho do CSV (posicional) | obrigatório |
| `--output` | Diretório de saída | `output` |
| `--artifacts-dir` | Diretório de artefatos treinados | `artifacts` |
| `--test-size` | Número de tickets de validação (balanceado) | `200` |
| `--with-justifications` | Gera justificativas na avaliação | `False` |
| `--justification` | Método de justificativa (`linear`/`llm`) | `linear` |
| `--llm` | Força justificativa por LLM | `False` |
| `--k-similar` | Tickets similares no RAG | `5` |
| `--model` | Override do modelo LLM | env var |
| `--reasoning` | Nível de reasoning (`low`/`medium`/`high`) | env var |
| `-v, --verbose` | Logs detalhados no terminal | `False` |

**Opções CLI (eval):**

| Opção | Descrição | Default |
|-------|-----------|---------|
| `ticket` | Texto do ticket (ou stdin) | obrigatório |
| `--artifacts-dir` | Diretório de artefatos treinados | `artifacts` |
| `--justification` | Método de justificativa (`linear`/`llm`) | `linear` |
| `--k-similar` | Tickets similares no RAG | `5` |
| `--model` | Override do modelo LLM | env var |
| `--reasoning` | Nível de reasoning (`low`/`medium`/`high`) | env var |
| `-v, --verbose` | Logs detalhados no terminal | `False` |

Observações:
- `train` sempre calcula métricas no conjunto de validação balanceado.
- `eval` imprime **apenas** o JSON exigido pelo desafio em stdout.

### Via Notebook

```bash
uv run jupyter notebook
# Abrir notebooks/main.ipynb (ou analysis/classificators conforme a necessidade)
```

## Output

Arquivos gerados em `output/`:

| Arquivo | Conteúdo |
|---------|----------|
| `classifications.json` | Report detalhado com `classifications` (inclui `classe` e `justificativa` quando habilitado) |
| `metrics.json` | Métricas no conjunto de validação |
| `run.log` | Log de execução |

Artefatos gerados em `artifacts/`:

| Arquivo | Conteúdo |
|---------|----------|
| `model.joblib` | Modelo treinado (TF-IDF + LinearSVC) |
| `metadata.json` | Metadados do treino |
| `rag/` | Embeddings + tickets para retrieval |

Se quiser um modelo pre-treinado, baixe [neste link](https://dhauzcom-my.sharepoint.com/:u:/g/personal/gabriel_soares_dhauz_com/IQCTWAU74JizQaXUMPVGf0yAASsL_2KO7kE8G3DNxjqEyJs?e=f7kghf).

## Dataset

Dataset público do Kaggle: [IT Service Ticket Classification](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)

- **~48.000 tickets** de suporte de TI
- **Classes:** extraídas do próprio CSV (ex.: Access, Administrative rights, Hardware, HR Support, Internal Project, Miscellaneous, Purchase, Storage)
