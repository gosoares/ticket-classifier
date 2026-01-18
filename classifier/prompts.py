"""Templates de prompts para classificação de tickets."""

SYSTEM_PROMPT = """Você é um classificador de tickets de suporte de TI.

Classifique o ticket em UMA das seguintes categorias:
{classes}

Responda APENAS com JSON no formato:
{{"classe": "<categoria>", "justificativa": "<explicação curta de 1-2 frases>"}}

A justificativa deve mencionar palavras-chave ou padrões do ticket que justificam a classificação."""


USER_PROMPT = """Classifique o seguinte ticket:

{ticket}

## Tickets Similares
{similar_tickets}

## Tickets de Referência (exemplos de cada classe)
{reference_tickets}"""


def format_similar_tickets(tickets: list[dict], max_chars: int = 200) -> str:
    """
    Formata tickets similares para o prompt.

    Args:
        tickets: Lista de dicts com 'text', 'class', 'score'
        max_chars: Máximo de caracteres por ticket

    Returns:
        String formatada com tickets numerados
    """
    if not tickets:
        return "(nenhum ticket similar encontrado)"

    lines = []
    for i, t in enumerate(tickets, 1):
        text = t["text"][:max_chars]
        if len(t["text"]) > max_chars:
            text += "..."
        lines.append(f"{i}. [{t['class']}] {text}")

    return "\n".join(lines)


def format_reference_tickets(
    representatives: dict[str, dict],
    exclude_classes: set[str] | None = None,
    max_chars: int = 150,
) -> str:
    """
    Formata tickets de referência para o prompt.

    Args:
        representatives: Dict[classe] -> {'text', 'class', 'score'}
        exclude_classes: Classes a omitir (já presentes nos similares)
        max_chars: Máximo de caracteres por ticket

    Returns:
        String formatada com um exemplo de cada classe
    """
    exclude = exclude_classes or set()
    lines = []

    for class_name in sorted(representatives.keys()):
        if class_name in exclude:
            continue

        t = representatives[class_name]
        text = t["text"][:max_chars]
        if len(t["text"]) > max_chars:
            text += "..."
        lines.append(f"- [{class_name}] {text}")

    if not lines:
        return "(todas as classes já representadas nos tickets similares)"

    return "\n".join(lines)
