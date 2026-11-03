Local AI Research Agent

Research, Analysis, Writing, Evaluation

Task 1.
Design a simple research agent to provide a summary report of web search results relating to a given user topic.
The agent has access to a duckduckgo web search tool to search for snippets relevant to the topic.
Research agent works in a loop until it decides no more tool calls (web searches) wanted. Then,
summarises research into a report.


Task 2.
Analyst Agent that interprets information, identifies patterns, extracts insights, and identifies gaps
in Research Agent's research. Returns analysis results in a valid pydantic model form (ResearchAnalysis)