"""LangGraph orchestration for the RAG justification pipeline."""

from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from classifier.config import K_SIMILAR, REASONING_EFFORT
from classifier.classifiers import Classifier
from classifier.llm import JustificationDetails, TicketJustifier
from classifier.logging_config import get_logger
from classifier.prompts import build_system_prompt, build_user_prompt
from classifier.rag import TicketRetriever

logger = get_logger("graph")


class PipelineState(TypedDict):
    """State flowing through the justification pipeline."""

    ticket: str
    predicted_class: str | None
    embedding: Any  # np.ndarray, but TypedDict doesn't support it well
    similar_tickets: list[dict]
    system_prompt: str
    user_prompt: str
    result: JustificationDetails | None


def create_graph(
    retriever: TicketRetriever,
    justifier: TicketJustifier,
    ml_classifier: Classifier | None = None,
    reasoning_effort: str | None = REASONING_EFFORT,
    k_similar: int = K_SIMILAR,
) -> CompiledStateGraph:
    """
    Create the justification pipeline graph.

    The pipeline has 4 nodes:
    - embed: Generate embedding for the ticket
    - retrieve: Find similar tickets using the embedding
    - classify_ml: Predict class using ML model (optional)
    - build_prompt: Construct system and user prompts
    - justify: Call LLM API and parse response

    Args:
        retriever: Indexed TicketRetriever instance
        justifier: TicketJustifier instance
        ml_classifier: Optional ML classifier (TF-IDF + LinearSVC)
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

    def classify_ml(state: PipelineState) -> dict:
        """Predict class using the ML classifier."""
        if ml_classifier is None:
            return {}
        predicted = ml_classifier.predict([state["ticket"]])[0]
        return {"predicted_class": predicted}

    def build_prompt(state: PipelineState) -> dict:
        """Build system and user prompts with RAG context."""
        logger.debug("Building prompts")
        predicted_class = state.get("predicted_class")
        if not predicted_class:
            raise ValueError("predicted_class is required to build the prompt")
        system = build_system_prompt()
        user = build_user_prompt(
            state["ticket"],
            predicted_class,
            state["similar_tickets"],
        )
        logger.debug(
            f"Prompts built: system={len(system)} chars, user={len(user)} chars"
        )
        return {"system_prompt": system, "user_prompt": user}

    def justify(state: PipelineState) -> dict:
        """Generate a justification using LLM."""
        logger.debug("Generating justification with LLM")
        result = justifier.call_llm(
            ticket=state["ticket"],
            predicted_class=state["predicted_class"],
            system_prompt=state["system_prompt"],
            user_prompt=state["user_prompt"],
            similar_tickets=state["similar_tickets"],
            reasoning_effort=reasoning_effort,
        )
        logger.debug("Justification complete")
        return {"result": result}

    graph = StateGraph(PipelineState)
    graph.add_node("embed", embed)
    graph.add_node("retrieve", retrieve)
    if ml_classifier is not None:
        graph.add_node("classify_ml", classify_ml)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("justify", justify)

    graph.add_edge(START, "embed")
    graph.add_edge("embed", "retrieve")
    if ml_classifier is not None:
        graph.add_edge("retrieve", "classify_ml")
        graph.add_edge("classify_ml", "build_prompt")
    else:
        graph.add_edge("retrieve", "build_prompt")
    graph.add_edge("build_prompt", "justify")
    graph.add_edge("justify", END)

    return graph.compile()


def justify_ticket(
    ticket: str,
    retriever: TicketRetriever,
    justifier: TicketJustifier,
    predicted_class: str | None = None,
    ml_classifier: Classifier | None = None,
    reasoning_effort: str | None = REASONING_EFFORT,
    k_similar: int = K_SIMILAR,
) -> JustificationDetails:
    """
    Generate a justification for a ticket using the RAG pipeline.

    Args:
        ticket: The ticket text to justify
        retriever: Indexed TicketRetriever instance
        justifier: TicketJustifier instance
        predicted_class: Class assigned by the ML classifier (optional if ml_classifier provided)
        ml_classifier: Optional ML classifier used to predict the class
        reasoning_effort: Reasoning effort level (low/medium/high) or None

    Returns:
        JustificationDetails with result, prompts, and metadata
    """
    logger.debug(f"Generating justification for ticket: {ticket[:50]}...")
    if predicted_class is None and ml_classifier is None:
        raise ValueError("Provide predicted_class or ml_classifier")

    graph = create_graph(
        retriever,
        justifier,
        ml_classifier=ml_classifier,
        reasoning_effort=reasoning_effort,
        k_similar=k_similar,
    )
    result = graph.invoke(
        {
            "ticket": ticket,
            "predicted_class": predicted_class,
            "embedding": np.array([]),
            "similar_tickets": [],
            "system_prompt": "",
            "user_prompt": "",
            "result": None,
        }
    )
    logger.debug("Justification generated")
    return result["result"]
