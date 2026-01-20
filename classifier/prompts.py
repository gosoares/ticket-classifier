"""Funções para geração de prompts de classificação de tickets."""


def build_system_prompt(classes: list[str]) -> str:
    """
    Gera o prompt do sistema para classificação.

    Args:
        classes: Lista de classes válidas para classificação

    Returns:
        Prompt formatado para o role "system"
    """
    classes_formatted = "\n".join(f"- {c}" for c in classes)

    return f"""Você é um classificador de tickets de suporte de TI.

Classifique o ticket em UMA das seguintes categorias:
{classes_formatted}

Responda APENAS com JSON no formato:
{{"classe": "<categoria>", "justificativa": "<explicação curta de 1-2 frases>"}}

A justificativa deve mencionar palavras-chave ou padrões do ticket que justificam a classificação."""


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


def _format_reference_tickets(
    representatives: dict[str, dict], exclude_classes: set[str] | None = None
) -> str:
    """
    Formata tickets de referência para o prompt.

    Args:
        representatives: Dict[classe] -> {'text', 'class', 'score'}
        exclude_classes: Classes a omitir (já presentes nos similares)

    Returns:
        String formatada com um exemplo de cada classe
    """
    exclude = exclude_classes or set()
    lines = []

    for class_name in sorted(representatives.keys()):
        if class_name in exclude:
            continue

        t = representatives[class_name]
        lines.append(f"- [{class_name}] {t['text']}")

    return "\n".join(lines)


def build_user_prompt(
    ticket: str,
    similar_tickets: list[dict],
    reference_tickets: dict[str, dict] | None = None,
) -> str:
    """
    Gera o prompt do usuário para classificação.

    A seção de tickets de referência só é incluída se reference_tickets
    for fornecido e não estiver vazio.

    Args:
        ticket: Texto do ticket a classificar
        similar_tickets: Lista de tickets similares do RAG
        reference_tickets: Dict de tickets representativos por classe (opcional)

    Returns:
        Prompt formatado para o role "user"
    """
    # Seção do ticket
    prompt = f"""Classifique o seguinte ticket:

{ticket}

## Tickets Similares
{_format_similar_tickets(similar_tickets)}"""

    # Seção de referência (opcional)
    if reference_tickets:
        # Determinar classes já representadas nos similares
        similar_classes = {t["class"] for t in similar_tickets}

        reference_text = _format_reference_tickets(
            reference_tickets, exclude_classes=similar_classes
        )

        # Só adiciona a seção se houver classes não representadas
        if reference_text:
            prompt += f"""

## Tickets de Referência (exemplos de cada classe)
{reference_text}"""

    return prompt
