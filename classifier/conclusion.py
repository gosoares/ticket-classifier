"""Geração de payload e prompts para conclusão automática."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np


def build_conclusion_system_prompt() -> str:
    """Retorna o prompt de sistema para o avaliador técnico."""
    return (
        "Você é um analista técnico rigoroso. Use somente o JSON fornecido. "
        "Não invente números, não presuma informações ausentes. "
        "Se algo estiver faltando, declare a lacuna e o impacto. "
        "Escreva em Português (Brasil), com foco em insights técnicos acionáveis."
    )


def build_conclusion_user_prompt(payload: dict[str, Any]) -> str:
    """Constrói o prompt do usuário com as tarefas e o payload."""
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""Gere um texto de análise dos resultados desta execução de um classificador de tickets com RAG, baseado exclusivamente no JSON abaixo.

Tarefas:
1) Resuma o desempenho global (interprete `accuracy`, `f1_macro`, `f1_weighted`, `cohen_kappa`, `mcc`) e o que isso sugere sobre generalização e consistência entre classes.
2) Analise desempenho por classe: destaque as melhores e piores (precision/recall/F1) e explique implicações práticas.
3) A partir de `top_confusoes` e da matriz de confusão, descreva os principais padrões de confusão (quem confunde com quem) e hipóteses prováveis.
4) Use as amostras `misclassificados` (se houver) para ilustrar padrões de erro recorrentes, citando evidências das justificativas e dos `similar_tickets` (classe e score).
5) Avalie o papel do RAG: quando parece ajudar vs. quando parece induzir erro (ex.: similares puxando para a classe errada).
6) Recomende 5 ações priorizadas (baixo esforço → alto impacto). Para cada uma: (a) o que mudar, (b) por que deve ajudar, (c) qual métrica/erro tende a melhorar.

Formato da resposta:
- Seções: `Resumo`, `Métricas Globais`, `Por Classe`, `Padrões de Confusão`, `Análise de Erros (amostras)`, `RAG`, `Ações Recomendadas`, `Conclusão`.
- Se `operacao.total_misclassificados_enviados` for 0, pule a seção `Análise de Erros (amostras)` e explique que não há evidências qualitativas nesta execução.

Regras:
- Não inclua números que não estejam no JSON.
- Quando citar uma métrica, use o valor exato do JSON.

```json
{payload_json}
```"""


def build_conclusion_payload(
    *,
    dataset: str,
    classes: list[str],
    test_size: int,
    k_similar: int,
    use_references: bool,
    embedding_model: str,
    llm_model: str | None,
    random_state: int,
    classifications: list[dict],
    errors: list[dict],
    metrics: dict | None,
    token_usage: Any = None,
    max_misclassified: int = 20,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Cria o payload para a conclusão automática."""
    timestamp = timestamp or datetime.now().isoformat()
    total = test_size
    classified = len(classifications)
    errors_count = len(errors)
    correct = sum(1 for c in classifications if c.get("correct"))
    wrong = classified - correct
    misclassified = [c for c in classifications if not c.get("correct")]
    misclassified_samples = misclassified[:max_misclassified]

    token_usage_dict = _normalize_token_usage(token_usage)
    avg_total_tokens = (
        token_usage_dict["total_tokens"] / classified if classified else 0
    )

    metrics_global = {
        "accuracy": _metric_value(metrics, "accuracy"),
        "f1_macro": _metric_value(metrics, "f1_macro"),
        "f1_weighted": _metric_value(metrics, "f1_weighted"),
        "cohen_kappa": _metric_value(metrics, "cohen_kappa"),
        "mcc": _metric_value(metrics, "mcc"),
    }
    per_class = metrics.get("per_class", {}) if metrics else {}
    confusion_matrix = _to_list(metrics.get("confusion_matrix")) if metrics else []
    confusion_matrix_norm = _to_list(
        metrics.get("confusion_matrix_normalized")
    ) if metrics else []
    if confusion_matrix and not confusion_matrix_norm:
        confusion_matrix_norm = _normalize_confusion_matrix(confusion_matrix)

    top_confusoes = (
        _compute_top_confusions(confusion_matrix, classes) if confusion_matrix else []
    )

    return {
        "execucao": {
            "timestamp": timestamp,
            "dataset": dataset,
            "classes": classes,
            "random_state": random_state,
            "test_size": test_size,
            "n_por_classe": (test_size // len(classes)) if classes else 0,
            "k_similar": k_similar,
            "use_references": use_references,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
        },
        "operacao": {
            "total": total,
            "classificados": classified,
            "erros": errors_count,
            "corretos": correct,
            "total_misclassificados": wrong,
            "total_misclassificados_enviados": len(misclassified_samples),
            "token_usage": {
                **token_usage_dict,
                "media_total_por_ticket": avg_total_tokens,
            },
            "retries_total": sum(c.get("retries", 0) for c in classifications),
        },
        "resultados": {
            "metricas_globais": metrics_global,
            "metricas_por_classe": per_class,
            "confusion_matrix": confusion_matrix,
            "confusion_matrix_normalized_row": confusion_matrix_norm,
            "top_confusoes": top_confusoes,
        },
        "amostras": {
            "misclassificados": [
                _format_misclassified_sample(item, idx)
                for idx, item in enumerate(misclassified_samples, 1)
            ]
        },
    }


def _metric_value(metrics: dict | None, key: str) -> float | None:
    if not metrics or metrics.get(key) is None:
        return None
    return float(metrics.get(key))


def _normalize_token_usage(token_usage: Any) -> dict[str, int]:
    if isinstance(token_usage, dict):
        return {
            "prompt_tokens": int(token_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(token_usage.get("completion_tokens", 0)),
            "total_tokens": int(token_usage.get("total_tokens", 0)),
        }
    if token_usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(token_usage, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(token_usage, "completion_tokens", 0)),
        "total_tokens": int(getattr(token_usage, "total_tokens", 0)),
    }


def _to_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _normalize_confusion_matrix(cm: list[list[int]]) -> list[list[float]]:
    normalized = []
    for row in cm:
        row_sum = sum(row)
        if row_sum == 0:
            normalized.append([0.0 for _ in row])
        else:
            normalized.append([float(val) / row_sum for val in row])
    return normalized


def _compute_top_confusions(
    cm: list[list[int]],
    classes: list[str],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    confusions = []
    for i, row in enumerate(cm):
        row_sum = sum(row)
        for j, count in enumerate(row):
            if i == j or count == 0:
                continue
            rate = float(count) / row_sum if row_sum else 0.0
            confusions.append(
                {
                    "true": classes[i] if i < len(classes) else str(i),
                    "pred": classes[j] if j < len(classes) else str(j),
                    "count": int(count),
                    "rate_in_true": rate,
                }
            )
    confusions.sort(
        key=lambda item: (
            -item["count"],
            -item["rate_in_true"],
            item["true"],
            item["pred"],
        )
    )
    return confusions[:top_n]


def _truncate_text(text: str, max_len: int) -> str:
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_misclassified_sample(sample: dict, idx: int) -> dict[str, Any]:
    similar_tickets = []
    for item in sample.get("similar_tickets", []):
        similar_tickets.append(
            {
                "class": item.get("class"),
                "score": item.get("score"),
                "text_trunc": _truncate_text(item.get("text", ""), 160),
            }
        )

    return {
        "id": idx,
        "ticket_trunc": _truncate_text(sample.get("ticket", ""), 240),
        "true_class": sample.get("true_class"),
        "predicted_class": sample.get("predicted_class"),
        "justificativa": sample.get("justification"),
        "similar_tickets": similar_tickets,
    }
