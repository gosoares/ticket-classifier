"""LangGraph orchestration for the classification + justification pipeline."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from classifier.classifiers import Classifier
from classifier.justifiers import Justifier
from classifier.logging_config import get_logger
from classifier.schemas import JustificationDetails

logger = get_logger("graph")


class PipelineState(TypedDict):
    """State flowing through the pipeline."""

    ticket: str
    predicted_class: str | None
    result: JustificationDetails | None


def create_graph(
    classifier: Classifier,
    justifier: Justifier,
) -> CompiledStateGraph:
    """
    Create the classification + justification pipeline graph.

    The pipeline has 2 nodes:
    - classify: Predict class using ML model
    - justify: Generate a justification (LLM or linear)

    Args:
        classifier: Trained classifier instance
        justifier: Justifier strategy (LLM or linear)

    Returns:
        Compiled StateGraph ready for invocation
    """

    def classify(state: PipelineState) -> dict:
        """Predict class using the classifier."""
        predicted = classifier.predict([state["ticket"]])[0]
        return {"predicted_class": predicted}

    def justify(state: PipelineState) -> dict:
        """Generate a justification using the configured strategy."""
        predicted_class = state.get("predicted_class")
        if not predicted_class:
            raise ValueError("predicted_class is required to justify")
        result = justifier.justify(
            ticket=state["ticket"],
            predicted_class=predicted_class,
        )
        return {"result": result}

    graph = StateGraph(PipelineState)
    graph.add_node("classify", classify)
    graph.add_node("justify", justify)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "justify")
    graph.add_edge("justify", END)

    return graph.compile()


def run_pipeline(
    graph: CompiledStateGraph,
    *,
    ticket: str,
) -> JustificationDetails:
    """Run a compiled pipeline for a single ticket."""
    result = graph.invoke(
        {
            "ticket": ticket,
            "predicted_class": None,
            "result": None,
        }
    )
    return result["result"]
