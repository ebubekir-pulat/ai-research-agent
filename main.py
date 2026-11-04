from graph import build_graph

# Note: available_models = ["qwen2.5:3b", "qwen2.5:7b", "llama3.1:8b"]

app = build_graph()

topic = input("Enter a research topic: ")

user_notes = input("Additional analysis instructions: ")

initial_state = {
    "topic": topic,
    "user_notes": user_notes,
    "research": "",
    "research_focus": "",
    "analysis": None,
    "research_rounds": 0
}

# Execute graph
result = app.invoke(initial_state)

print("\n===== GRAPH STATE =====")

print("Research rounds:", result["research_rounds"])
print("Research length:", len(result["research"]))
print("Research focus:", result["research_focus"])

analysis = result["analysis"]

print("\n===== FINAL RESEARCH ANALYSIS =====")

print("\nKey findings:")
for finding in analysis.key_findings:
    print("-", finding)

print("\nCommon themes:")
for theme in analysis.common_themes:
    print("-", theme)

print("\nKnowledge gaps:")
for gap in analysis.knowledge_gaps:
    print("-", gap)

print("\nRecommendations:")
for recommendation in analysis.recommendations:
    print("-", recommendation)

print("\nOverall assessment:")
print(analysis.overall_assessment)