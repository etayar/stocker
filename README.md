# Stocker

Stocker is a model-agnostic orchestration system that turns a user prompt into an actionable, reproducible result by:
1) planning which components to run (tools, RAG, models),
2) executing them with logging + caching,
3) returning a grounded answer backed by structured tool outputs and citations when available.

Primary goal: decision support via measurable signals (e.g., sentiment, risk flags, trend scores) that can be stored, compared over time, and used downstream.
