"""Funções para geração de prompts de justificativa de tickets."""


def build_system_prompt() -> str:
    """
    Gera o prompt do sistema para justificativa.

    Returns:
        Prompt formatado para o role "system"
    """
    return """Você é um assistente que gera justificativas para classificações de tickets de suporte de TI.

Responda APENAS com JSON no formato:
{"justificativa": "<explicação curta e objetiva de 1-3 frases>"}

IMPORTANTE: A justificativa deve ser escrita em **Português (Brasil)**.
Use evidências do ticket e dos exemplos fornecidos para sustentar a classe informada.
Não mencione tickets similares explicitamente (ex.: “ticket 2”)."""


def _format_similar_tickets(tickets: list[dict]) -> str:
    """
    Formata tickets similares para o prompt.

    Args:
        tickets: Lista de dicts com 'text', 'class', 'score'

    Returns:
        String formatada com tickets numerados
    """
    if not tickets:
        return "(nenhum ticket similar encontrado)"

    lines = []
    for i, t in enumerate(tickets, 1):
        lines.append(f"{i}. [{t['class']}] {t['text']}")

    return "\n".join(lines)


def build_user_prompt(
    ticket: str,
    predicted_class: str,
    similar_tickets: list[dict],
) -> str:
    """
    Gera o prompt do usuário para justificativa.

    Args:
        ticket: Texto do ticket a classificar
        predicted_class: Classe já definida pelo classificador
        similar_tickets: Lista de tickets similares do RAG

    Returns:
        Prompt formatado para o role "user"
    """
    prompt = f"""Gere uma justificativa para a classificação abaixo.

**Classe atribuída:** {predicted_class}

**Ticket:**
{ticket}

## Tickets Similares (como evidência)
{_format_similar_tickets(similar_tickets)}

Responda APENAS com JSON no formato:
{{"justificativa": "<explicação curta e objetiva de 2-4 frases>"}}

Regras:
- A justificativa deve ser auto contida.
- Não mencione tickets similares explicitamente (ex.: “ticket 2”).
- Use apenas evidências gerais do conteúdo do ticket."""

    return prompt
