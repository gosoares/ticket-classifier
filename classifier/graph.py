"""LangGraph orchestration for the RAG classification pipeline."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from classifier.config import K_SIMILAR
from classifier.llm import ClassificationResult, TicketClassifier
from classifier.rag import TicketRetriever


class PipelineState(TypedDict):
    """State flowing through the classification pipeline."""

    ticket: str
    similar_tickets: list[dict]
    result: ClassificationResult | None


def create_graph(
    retriever: TicketRetriever,
    classifier: TicketClassifier,
    classes: list[str],
    reference_tickets: dict[str, dict] | None = None,
) -> CompiledStateGraph:
    """
    Create the classification pipeline graph.

    Args:
        retriever: Indexed TicketRetriever instance
        classifier: TicketClassifier instance
        classes: List of valid class names
        reference_tickets: Optional dict of representative tickets per class

    Returns:
        Compiled StateGraph ready for invocation
    """

    def retrieve(state: PipelineState) -> dict:
        """Retrieve similar tickets from the index."""
        similar = retriever.retrieve(state["ticket"], k=K_SIMILAR)
        return {"similar_tickets": similar}

    def classify(state: PipelineState) -> dict:
        """Classify the ticket using LLM with RAG context."""
        result = classifier.classify(
            ticket=state["ticket"],
            similar_tickets=state["similar_tickets"],
            classes=classes,
            reference_tickets=reference_tickets,
        )
        return {"result": result}

    graph = StateGraph(PipelineState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("classify", classify)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "classify")
    graph.add_edge("classify", END)

    return graph.compile()


def classify_ticket(
    ticket: str,
    retriever: TicketRetriever,
    classifier: TicketClassifier,
    classes: list[str],
    reference_tickets: dict[str, dict] | None = None,
) -> ClassificationResult:
    """
    Classify a single ticket using the RAG pipeline.

    Args:
        ticket: The ticket text to classify
        retriever: Indexed TicketRetriever instance
        classifier: TicketClassifier instance
        classes: List of valid class names
        reference_tickets: Optional dict of representative tickets per class

    Returns:
        ClassificationResult with classe and justificativa
    """
    graph = create_graph(retriever, classifier, classes, reference_tickets)
    result = graph.invoke({"ticket": ticket, "similar_tickets": [], "result": None})
    return result["result"]
