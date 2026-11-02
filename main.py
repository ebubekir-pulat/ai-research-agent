
from agents.researcher import ResearchAgent

agent = ResearchAgent()

topic = input("Enter a research topic: ")

result = agent.run(topic)

print("\n===== RESEARCH REPORT =====\n")
print(result)
