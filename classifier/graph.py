"""LangGraph orchestration for the RAG classification pipeline."""

from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from classifier.config import K_SIMILAR, REASONING_EFFORT
from classifier.llm import ClassificationDetails, TicketClassifier
from classifier.logging_config import get_logger
from classifier.prompts import build_system_prompt, build_user_prompt
from classifier.rag import TicketRetriever

logger = get_logger("graph")


class PipelineState(TypedDict):
    """State flowing through the classification pipeline."""

    ticket: str
    embedding: Any  # np.ndarray, but TypedDict doesn't support it well
    similar_tickets: list[dict]
    system_prompt: str
    user_prompt: str
    result: ClassificationDetails | None


def create_graph(
    retriever: TicketRetriever,
    classifier: TicketClassifier,
    classes: list[str],
    reference_tickets: dict[str, dict] | None = None,
    reasoning_effort: str | None = REASONING_EFFORT,
    k_similar: int = K_SIMILAR,
) -> CompiledStateGraph:
    """
    Create the classification pipeline graph.

    The pipeline has 4 nodes:
    - embed: Generate embedding for the ticket
    - retrieve: Find similar tickets using the embedding
    - build_prompt: Construct system and user prompts
    - classify: Call LLM API and parse response

    Args:
        retriever: Indexed TicketRetriever instance
        classifier: TicketClassifier instance
        classes: List of valid class names
        reference_tickets: Optional dict of representative tickets per class
        reasoning_effort: Reasoning effort level (low/medium/high) or None

    Returns:
        Compiled StateGraph ready for invocation
    """

    def embed(state: PipelineState) -> dict:
        """Generate embedding for the ticket."""
        logger.debug("Generating embedding for ticket")
        embedding = retriever.embed(state["ticket"])
        logger.debug(f"Embedding generated: {embedding.shape[0]} dimensions")
        return {"embedding": embedding}

    def retrieve(state: PipelineState) -> dict:
        """Retrieve similar tickets from the index."""
        logger.debug("Retrieving similar tickets")
        similar = retriever.search(state["embedding"], k=k_similar)
        logger.debug(f"Retrieved {len(similar)} similar tickets")
        return {"similar_tickets": similar}

    def build_prompt(state: PipelineState) -> dict:
        """Build system and user prompts with RAG context."""
        logger.debug("Building prompts")
        system = build_system_prompt(classes)
        user = build_user_prompt(
            state["ticket"], state["similar_tickets"], reference_tickets
        )
        logger.debug(
            f"Prompts built: system={len(system)} chars, user={len(user)} chars"
        )
        return {"system_prompt": system, "user_prompt": user}

    def classify(state: PipelineState) -> dict:
        """Classify the ticket using LLM."""
        logger.debug("Classifying ticket with LLM")
        result = classifier.call_llm(
            ticket=state["ticket"],
            system_prompt=state["system_prompt"],
            user_prompt=state["user_prompt"],
            similar_tickets=state["similar_tickets"],
            valid_classes=classes,
            reasoning_effort=reasoning_effort,
        )
        logger.debug(f"Classification complete: {result.result.classe}")
        return {"result": result}

    graph = StateGraph(PipelineState)
    graph.add_node("embed", embed)
    graph.add_node("retrieve", retrieve)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("classify", classify)

    graph.add_edge(START, "embed")
    graph.add_edge("embed", "retrieve")
    graph.add_edge("retrieve", "build_prompt")
    graph.add_edge("build_prompt", "classify")
    graph.add_edge("classify", END)

    return graph.compile()


def classify_ticket(
    ticket: str,
    retriever: TicketRetriever,
    classifier: TicketClassifier,
    classes: list[str],
    reference_tickets: dict[str, dict] | None = None,
    reasoning_effort: str | None = REASONING_EFFORT,
    k_similar: int = K_SIMILAR,
) -> ClassificationDetails:
    """
    Classify a single ticket using the RAG pipeline.

    Args:
        ticket: The ticket text to classify
        retriever: Indexed TicketRetriever instance
        classifier: TicketClassifier instance
        classes: List of valid class names
        reference_tickets: Optional dict of representative tickets per class
        reasoning_effort: Reasoning effort level (low/medium/high) or None

    Returns:
        ClassificationDetails with result, prompts, and metadata
    """
    logger.debug(f"Classifying ticket: {ticket[:50]}...")
    graph = create_graph(
        retriever,
        classifier,
        classes,
        reference_tickets,
        reasoning_effort,
        k_similar=k_similar,
    )
    result = graph.invoke(
        {
            "ticket": ticket,
            "embedding": np.array([]),
            "similar_tickets": [],
            "system_prompt": "",
            "user_prompt": "",
            "result": None,
        }
    )
    logger.debug(f"Result: {result['result'].result.classe}")
    return result["result"]
