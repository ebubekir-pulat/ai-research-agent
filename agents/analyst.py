from ollama import chat, ChatResponse
from pydantic import BaseModel


class ResearchAnalysis(BaseModel):
    key_findings: list[str]
    common_themes: list[str]
    conflicting_information: list[str]
    knowledge_gaps: list[str]
    overall_assessment: str


class AnalystAgent:

    def __init__(self, model="llama3.1:8b"):
        self.model = model

    def run(self, topic: str, research: str) -> ResearchAnalysis:

        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an expert research analyst.

                    Analyse the research provided.

                    Your responsibilities:
                    1. Extract important findings.
                    2. Identify common themes.
                    3. Identify conflicting information.
                    4. Identify missing information.
                    5. Produce an overall assessment.

                    Only use information supported by the research.
                    Do not invent facts.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Research topic:
                    {topic}

                    Research findings:
                    {research}
                    """
                }
            ],
            format=ResearchAnalysis.model_json_schema()
        )

        return ResearchAnalysis.model_validate_json(
            response.message.content
        )