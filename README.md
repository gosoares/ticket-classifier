# IT Service Ticket Classification

Sistema de classificação automática de tickets de suporte de TI com ML (TF-IDF + LinearSVC) e geração de **justificativas**:

- **Linear (padrão):** determinístico, rápido e sem dependência de LLM
- **LLM + RAG (opcional):** justificativas em linguagem natural condicionadas a evidências recuperadas

## Objetivo

- **Entrada:** texto do ticket (string)
- **Saída:** `{"classe": "...", "justificativa": "..."}` (classe vem do ML; justificativa pode ser linear ou via LLM)

O sistema classifica tickets em 8 categorias e fornece uma justificativa explicando o motivo da classificação.

## Notebooks

- `notebooks/analysis.ipynb`: análise exploratória (EDA).
- `notebooks/classificators.ipynb`: testes de classificadores (TF-IDF, embeddings, RAG) com métricas no conjunto de teste.
- `notebooks/main.ipynb`: visão geral, prompts, pipeline e avaliação final (usa o classificador escolhido: **TF-IDF + LinearSVC**).

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

### Via CLI

```bash
# Execução padrão (justificativa linear, sem LLM)
uv run python main.py

# Ver todas as opções
uv run python main.py --help

# Execução com parâmetros customizados
uv run python main.py -v --test-size 200 --k-similar 5

# Usar LLM para justificativa (requer .env configurado)
uv run python main.py --justification llm

# Override do modelo LLM (opcional)
uv run python main.py --justification llm --model "gpt-4o-mini"
```

**Opções CLI:**

| Opção | Descrição | Default |
|-------|-----------|---------|
| `--dataset` | Caminho do CSV | `dataset.csv` |
| `--output` | Diretório de saída | `output` |
| `--test-size` | Número de tickets de validação (balanceado) | `200` |
| `--k-similar` | Tickets similares no RAG | `5` |
| `--justification` | Método de justificativa (`linear`/`llm`) | `linear` |
| `--model` | Override do modelo LLM | env var |
| `--reasoning` | Nível de reasoning (`low`/`medium`/`high`) | env var |
| `-v, --verbose` | Logs detalhados no terminal | `False` |

Observação: apesar do nome `--test-size`, o pipeline atual separa o dataset em **treino/teste/validação** e usa o conjunto
de **validação** (balanceado) para a avaliação final e geração de justificativas.

### Via Notebook

```bash
uv run jupyter notebook
# Abrir notebooks/main.ipynb (ou analysis/classificators conforme a necessidade)
```

## Output

Arquivos gerados em `output/`:

| Arquivo | Conteúdo |
|---------|----------|
| `classifications.json` | Classificações detalhadas com justificativas (linear ou LLM) |
| `metrics.json` | Métricas no conjunto de validação |
| `run.log` | Log de execução |

## Dataset

Dataset público do Kaggle: [IT Service Ticket Classification](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)

- **~48.000 tickets** de suporte de TI
- **Classes:** extraídas do próprio CSV (ex.: Access, Administrative rights, Hardware, HR Support, Internal Project, Miscellaneous, Purchase, Storage)
