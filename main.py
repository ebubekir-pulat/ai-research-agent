from graph import build_graph
from pathlib import Path
from datetime import datetime

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
    "research_rounds": 0,
    "report": ""
}

# Execute graph
result = app.invoke(initial_state)

print("\n===== FINAL RESEARCH REPORT =====\n")
print(result["report"])


report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)

filename = datetime.now().strftime("report_%Y%m%d_%H%M%S.md")
report_path = report_dir / filename

report_path.write_text(result["report"], encoding="utf-8")

print(f"\nReport saved to: {report_path}")