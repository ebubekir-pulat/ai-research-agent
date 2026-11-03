from agents.researcher import ResearchAgent

available_models = ["qwen2.5:3b", "qwen2.5:7b", "llama3.1:8b"]

agent = ResearchAgent(model=available_models[2])

topic = input("Enter a research topic: ")

result = agent.run(topic)

print("\n===== RESEARCH REPORT =====\n")
print(result)