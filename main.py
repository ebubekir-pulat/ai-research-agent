from agents.researcher import ResearchAgent
from agents.analyst import AnalystAgent

available_models = ["qwen2.5:3b", "qwen2.5:7b", "llama3.1:8b"]
model = available_models[2]

researcher = ResearchAgent(model=model)
analyst = AnalystAgent(model=model)

topic = input("Enter a research topic: ")

# Stage 1: Research
research = researcher.run(topic)

# Stage 2: Analysis
analysis = analyst.run(topic, research)

print("\n===== RESEARCH ANALYSIS =====")

print("\nKey findings:")
for finding in analysis.key_findings:
    print("-", finding)

print("\nCommon themes:")
for theme in analysis.common_themes:
    print("-", theme)

print("\nConflicting information:")
for conflict in analysis.conflicting_information:
    print("-", conflict)

print("\nKnowledge gaps:")
for gap in analysis.knowledge_gaps:
    print("-", gap)

print("\nOverall assessment:")
print(analysis.overall_assessment)