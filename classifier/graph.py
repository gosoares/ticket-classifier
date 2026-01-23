"""LangGraph pipeline for ticket evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from classifier.justifiers import Justifier
from classifier.rag import TicketRetriever


class EvalState(TypedDict, total=False):
    ticket: str
    vector: Any
    predicted_class: str
    similar_tickets: list[dict]
    justificativa: str
    justification_source: str
    evidence_terms: list[str] | None
    output: dict


@dataclass(slots=True)
class GraphRuntime:
    classifier: Any
    justifier: Justifier
    justification_mode: str
    retriever: TicketRetriever | None
    k_similar: int


def build_eval_graph(runtime: GraphRuntime):
    """Build and compile the LangGraph for ticket evaluation."""
    graph = StateGraph(EvalState)

    def vectorize(state: EvalState) -> EvalState:
        ticket = state["ticket"]
        vector = runtime.classifier.features.transform([ticket])
        return {"vector": vector}

    def classify(state: EvalState) -> EvalState:
        predicted_class = runtime.classifier.classifier.predict(state["vector"])[0]
        return {"predicted_class": predicted_class}

    def retrieve(state: EvalState) -> EvalState:
        if runtime.justification_mode != "llm":
            return {"similar_tickets": []}
        if runtime.retriever is None:
            raise ValueError("Retriever not available for LLM justifications.")
        similar = runtime.retriever.retrieve(state["ticket"], k=runtime.k_similar)
        return {"similar_tickets": similar}

    def justify(state: EvalState) -> EvalState:
        details = runtime.justifier.justify(
            ticket=state["ticket"],
            predicted_class=state["predicted_class"],
            similar_tickets=state.get("similar_tickets"),
        )
        return {
            "justificativa": details.result.justificativa,
            "justification_source": details.justification_source,
            "evidence_terms": details.evidence_terms,
            "similar_tickets": details.similar_tickets,
        }

    def format_output(state: EvalState) -> EvalState:
        return {
            "output": {
                "classe": state["predicted_class"],
                "justificativa": state["justificativa"],
            }
        }

    graph.add_node("vectorize", vectorize)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("justify", justify)
    graph.add_node("format_output", format_output)

    graph.set_entry_point("vectorize")
    graph.add_edge("vectorize", "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "justify")
    graph.add_edge("justify", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()
