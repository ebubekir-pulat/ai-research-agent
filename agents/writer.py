from ollama import chat, ChatResponse
from agents.analyst import ResearchAnalysis


class WriterAgent:
    def __init__(self, model="llama3.1:8b"):
        self.model = model

    def run(
        self,
        topic: str,
        research: str,
        analysis: ResearchAnalysis,
        user_notes: str = ""
    ) -> str:

        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a senior technical report writer.

                    Create a clear, professional research report.

                    Structure:
                    1. Executive Summary
                    2. Research Scope
                    3. Key Findings
                    4. Comparative Analysis
                    5. Practical Recommendations
                    6. Limitations and Knowledge Gaps
                    7. Conclusion

                    Requirements:
                    - Use Markdown headings and tables where appropriate.
                    - Prioritise technical accuracy.
                    - Distinguish verified findings from assumptions.
                    - Do not invent specifications, benchmarks or sources.
                    - Explicitly identify insufficient evidence.
                    - Make recommendations practical and actionable.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    RESEARCH TOPIC:
                    {topic}

                    RAW RESEARCH:
                    {research}

                    ANALYST FINDINGS:
                    {analysis.model_dump_json(indent=2)}

                    USER INSTRUCTIONS:
                    {user_notes}
                    """
                }
            ]
        )

        return response.message.content