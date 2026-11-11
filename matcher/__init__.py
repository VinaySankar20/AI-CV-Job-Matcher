"""CV ↔ Job Matcher: local AI scoring and cover letters with LangGraph, LangChain and Ollama."""
import os

# Privacy: never send traces (which would contain your CV) to LangSmith or any other cloud service.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
