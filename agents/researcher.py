import json
from ollama import chat
from tools.search import web_search


class ResearchAgent:

    def __init__(self, model="qwen2.5:3b", max_iterations=10):
        self.model = model
        self.max_iterations = max_iterations

    def run(self, topic: str):

        messages = [
            {
                "role": "system",
                "content": """
                You are an autonomous research agent.

                Your job is to thoroughly research the user's topic
                using the available web_search tool.

                Research instructions:

                1. Break complex topics into smaller research questions.
                2. Make web search queries fine-scoped to improve quality of results.
                3. Use multiple web searches when necessary.
                4. Search different aspects of the topic.
                5. Do not rely on a single search if more information
                is needed.
                6. Analyse the results returned by web_search.
                7. If information is missing, perform additional searches.
                8. Once you have enough information, provide a
                comprehensive and structured summary.

                You can call web_search multiple times.

                Do not provide your final answer until you have
                completed your research.
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
                    "description": """
                    Search the internet for information about a specific query.
                    Use different queries to investigate different aspects
                    of a research topic.
                    """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # Agent loop
        for iteration in range(self.max_iterations):
            print(f"\n--- Agent iteration {iteration + 1} ---")

            response = chat(
                model=self.model,
                messages=messages,
                tools=tools
            )

            messages.append(response.message)
        
            tool_calls = response.message.tool_calls or []

            print(f"Tool calls requested: {len(tool_calls)}")

            # No more tool calls
            if not tool_calls:
                break

            # Execute every requested tool call
            for tool_call in tool_calls:

                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments

                if tool_name == "web_search":

                    query = arguments["query"]

                    print(f"\nSearching: {query}")

                    results = web_search(query)

                    print(f"Results found: {len(results)}")

                    # Send tool results back to the model
                    messages.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": json.dumps(results)
                    })


        print("\nResearch completed.")

        # Ask model to summarise available research
        final_response = chat(
            model=self.model,
            messages=messages + [
                {
                    "role": "user",
                    "content": """
                    You have reached the research iteration limit.
                    Summarise all the research collected so far.
                    Do not request additional searches.
                    """
                }
            ]
        )

        return final_response.message.content