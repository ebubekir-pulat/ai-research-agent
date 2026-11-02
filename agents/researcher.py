
from ollama import chat, ChatResponse
from pydantic import BaseModel


class ResearchPlan(BaseModel):
    topic: str
    research_questions: list[str]
    objective: str


class ResearchAgent:

    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    def run(self, topic: str) -> ResearchPlan:

        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an expert research agent.

                    Given a topic:
                    1. Identify the research objective.
                    2. Generate exactly 3 useful research questions.
                    3. Make questions specific and actionable.
                    """
                },
                {
                    "role": "user",
                    "content": f"Research topic: {topic}"
                }
            ],
            format=ResearchPlan.model_json_schema()
        )

        return ResearchPlan.model_validate_json(
            response.message.content
        )
