
from agents.researcher import ResearchAgent


agent = ResearchAgent()

topic = input("Enter a topic: ")

result = agent.run(topic)

print("\nResearch objective:")
print(result.objective)

print("\nResearch questions:")

for question in result.research_questions:
    print(f"- {question}")