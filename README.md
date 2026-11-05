Local Research Agent

A multi-agent AI research system built with Python, Ollama, LangGraph, and scikit-learn.

Features

Research Agent: Searches the web and gathers information.

ML Classifier: Uses TF-IDF and Logistic Regression to categorise research findings.

Analyst Agent: Identifies key findings, themes, knowledge gaps, and recommendations.

Writer Agent: Generates a structured Markdown research report.

LangGraph: Orchestrates the agents and manages iterative research.

Architecture

Research Agent → ML Classifier → Analyst Agent → Writer Agent → Markdown Report

Technologies

Python

Ollama

LangGraph

scikit-learn

TF-IDF

Logistic Regression

DuckDuckGo Search

Run locally

Install dependencies:

pip install -r requirements.txt

Make sure Ollama is running and the required model is available:

ollama pull llama3.1:8b

Run the project:

python3 main.py

Generated reports are saved in the reports/ directory.

ML component

The classifier uses a small labelled dataset to categorize research text into:

Hardware

Tool calling

Performance

Agentic workflows

It uses a scikit-learn Pipeline containing TF-IDF vectorization and Logistic Regression.

Limitations

The training dataset is intentionally small for this learning project. Classification predictions and confidence scores are experimental and should not be treated as verified facts.