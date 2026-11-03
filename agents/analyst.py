from ollama import chat, ChatResponse
from pydantic import BaseModel


class ResearchAnalysis(BaseModel):
    key_findings: list[str]
    common_themes: list[str]
    conflicting_information: list[str]
    knowledge_gaps: list[str]
    recommendations: list[str]
    overall_assessment: str


class AnalystAgent:

    def __init__(self, model="llama3.1:8b"):
        self.model = model

    def run(self, topic: str, research: str, user_notes: str = "") -> ResearchAnalysis:

        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a senior technical research analyst.

                    Your job is to critically analyse research findings
                    and produce evidence-based conclusions.

                    Follow these instructions:

                    1. KEY FINDINGS
                    - Extract specific, factual findings.
                    - Include relevant technical details.
                    - Avoid generic statements.
                    - Do not simply repeat the research topic.

                    2. COMMON THEMES
                    - Identify patterns across different sources.
                    - Explain why these patterns matter.

                    3. CONFLICTING INFORMATION
                    - Identify genuinely contradictory claims.
                    - Explain the nature of the disagreement.
                    - Do not list different technologies as conflicts.

                    4. KNOWLEDGE GAPS
                    - Identify important information missing
                        from the research.
                    - Consider missing benchmarks, specifications,
                        comparisons and supporting evidence.

                    5. RECOMMENDATIONS AND OVERALL ASSESSMENT
                    - Compare the available ideas/options.
                    - Discuss strengths and weaknesses.
                    - Provide practical conclusions.
                    - Explain where evidence is insufficient.

                    IMPORTANT RULES:

                    - Only make claims supported by the research.
                    - Never invent benchmarks, specifications or capabilities.
                    - Distinguish facts from assumptions.
                    - If evidence is insufficient, explicitly say so.
                    - Avoid vague statements such as
                    "the technology is diverse and evolving".
                    - Prioritise technical depth over generic summaries.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    RESEARCH TOPIC:
                    {topic}

                    RESEARCH FINDINGS:
                    {research}

                    {user_notes}
                    """
                }
            ],
            format=ResearchAnalysis.model_json_schema()
        )

        return ResearchAnalysis.model_validate_json(
            response.message.content
        )