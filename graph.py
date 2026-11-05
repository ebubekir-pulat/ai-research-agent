from typing import TypedDict, Literal
from agents.researcher import ResearchAgent
from agents.analyst import ResearchAnalysis, AnalystAgent
from agents.writer import WriterAgent
from langgraph.graph import StateGraph, START, END
from ml.classifier import ResearchClassifier
from collections import Counter
import re
from pathlib import Path
from datetime import datetime

class ResearchState(TypedDict):
    topic: str
    user_notes: str
    research: str
    research_focus: str
    research_categories: list[dict]
    analysis: ResearchAnalysis | None
    research_rounds: int
    report:str


researcher = ResearchAgent(model="llama3.1:8b")
analyst = AnalystAgent(model="llama3.1:8b")
writer = WriterAgent(model="llama3.1:8b")
classifier = ResearchClassifier()
classifier.train()


def research_node(state: ResearchState) -> dict:
    print("\n===== RESEARCH NODE =====")

    focus = state.get("research_focus", "")

    research = researcher.run(state["topic"], focus)

    existing_research = state.get("research", "")

    combined_research = (
        existing_research
        + "\n\n"
        + research
    ).strip()

    return {
        "research": combined_research,
        "research_rounds": state["research_rounds"] + 1
    }


def classify_node(state: ResearchState) -> dict:
    print("\n===== ML CLASSIFICATION NODE =====")

    # Split research into individual sentences
    sentences = re.split(r'(?<=[.!?])\s+', state["research"])

    classified = []

    for sentence in sentences:
        sentence = sentence.strip("- •\t ")

        if len(sentence) > 20:
            result = classifier.predict(sentence)

            classified.append({
                "text": sentence,
                **result
            })

    counts = Counter(
        item["category"] for item in classified
    )

    print("ML category counts:", dict(counts))

    return {"research_categories": classified}


def analysis_node(state: ResearchState) -> dict:

    print("\n===== ANALYSIS NODE =====")

    classifications = "\n".join(
        f"- {item['category']}: {item['text']}"
        for item in state.get("research_categories", [])
    )

    analysis = analyst.run(
        topic=state["topic"],
        research=(
            state["research"]
            + "\n\nML CLASSIFICATION HINTS "
            "(automatically generated; may be inaccurate):\n"
            + classifications
        ),
        user_notes=state["user_notes"]
    )

    return {
        "analysis": analysis,
        "research_focus": "\n".join(
            analysis.knowledge_gaps
        )
    }


def should_research_more(
    state: ResearchState
) -> Literal["research", "write"]:

    analysis = state["analysis"]

    # Maximum of two research rounds
    if state["research_rounds"] >= 2:
        return "write"

    if analysis and analysis.knowledge_gaps:
        print("\nKnowledge gaps identified.")
        print("Returning to Research Agent...")
        return "research"

    return "write"


def writer_node(state: ResearchState) -> dict:
    print("\n===== WRITER NODE =====")

    report = writer.run(
        topic=state["topic"],
        research=state["research"],
        analysis=state["analysis"],
        user_notes=state["user_notes"]
    )

    return {"report": report}


def build_graph():
    workflow = StateGraph(state_schema=ResearchState)

    # Register nodes
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("classify", classify_node)

    # Define workflow
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "classify")
    workflow.add_edge("classify", "analysis")

    # Conditional routing
    workflow.add_conditional_edges(
        "analysis",
        should_research_more,
        {
            "research": "research",
            "write": "writer"
        }
    )

    workflow.add_edge("writer", END)

    # Compile
    return workflow.compile()