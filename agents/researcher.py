
import json
from ollama import chat
from tools.search import web_search


class ResearchAgent:

    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    def run(self, topic: str):

        messages = [
            {
                "role": "system",
                "content": """
                You are a research agent.

                You have access to a web search tool.
                Use it to find relevant information about
                the user's research topic.

                After searching, summarise the findings.
                """
            },
            {
                "role": "user",
                "content": f"Research topic: {topic}"
            }
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # Ask the model what to do
        response = chat(
            model=self.model,
            messages=messages,
            tools=tools
        )

        messages.append(response.message)

        # Execute requested tools
        for tool_call in response.message.tool_calls or []:

            if tool_call.function.name == "web_search":

                query = tool_call.function.arguments["query"]

                print(f"\nSearching: {query}")

                results = web_search(query)

                messages.append({
                    "role": "tool",
                    "tool_name": tool_call.function.name,
                    "content": json.dumps(results)
                })

        # Ask model to produce final answer
        final_response = chat(
            model=self.model,
            messages=messages
        )

        return final_response.message.content
