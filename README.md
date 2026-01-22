# IT Service Ticket Classification

Sistema de classificação automática de tickets de suporte de TI com ML (TF-IDF + LinearSVC) e RAG para **justificativas** via LLM.

## Objetivo

- **Entrada:** texto do ticket (string)
- **Saída:** `{"classe": "...", "justificativa": "..."}` (classe vem do ML, justificativa do LLM)

O sistema classifica tickets em 8 categorias e fornece uma justificativa explicando o motivo da classificação.

## Notebooks

- `notebooks/analysis.ipynb`: análise exploratória (EDA).
- `notebooks/classificators.ipynb`: testes de classificadores (TF-IDF, embeddings, RAG) com métricas no conjunto de teste.
- `notebooks/main.ipynb`: visão geral, prompts, pipeline e avaliação final (usa o classificador escolhido: **TF-IDF + LinearSVC**).

## Requisitos

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Acesso a API LLM compatível com OpenAI

## Instalação

```bash
git clone <repo>
cd ticket-classifier
uv sync
```

## Configuração

### 1. Criar arquivo `.env`

```bash
cp .env.example .env
```

### 2. Configurar variáveis de ambiente

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `LLM_BASE_URL` | URL da API | `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | Chave da API | `sk-...` |
| `LLM_MODEL` | Nome do modelo | `xiaomi/mimo-v2-flash:free` |
| `LLM_REASONING_EFFORT` | Nível de reasoning (padrão: `medium`) | `low`, `medium`, `high`, ou vazio para desativar |

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
LLM_MODEL=qwen2.5:7b
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
# Execução padrão (reasoning habilitado)
uv run python main.py

# Ver todas as opções
uv run python main.py --help

# Execução com parâmetros customizados
uv run python main.py -v --test-size 200 --k-similar 5

# Desabilitar reasoning para menor uso de tokens
uv run python main.py --reasoning ""
```

**Opções CLI:**

| Opção | Descrição | Default |
|-------|-----------|---------|
| `--dataset` | Caminho do CSV | `dataset.csv` |
| `--output` | Diretório de saída | `output` |
| `--test-size` | Número de tickets de teste | `200` |
| `--k-similar` | Tickets similares no RAG | `5` |
| `--model` | Override do modelo LLM | env var |
| `--reasoning` | Nível de reasoning (`low`/`medium`/`high`) | `medium` |
| `-v, --verbose` | Logs detalhados no terminal | `False` |

### Via Notebook

```bash
uv run jupyter notebook
# Abrir notebooks/main.ipynb (ou analysis/classificators conforme a necessidade)
```

## Output

Arquivos gerados em `output/`:

| Arquivo | Conteúdo |
|---------|----------|
| `classifications.json` | Classificações detalhadas com justificativas e reasoning |
| `metrics.json` | Métricas de avaliação |
| `run.log` | Log de execução |

## Dataset

Dataset público do Kaggle: [IT Service Ticket Classification](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)

- **~48.000 tickets** de suporte de TI
- **8 classes:** Access/Login, Administrative rights, Hardware, HR Support, Internal Project, Miscellaneous, Purchase, Software
