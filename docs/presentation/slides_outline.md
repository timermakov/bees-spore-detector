# 10-Minute Presentation Outline

## Slide 1 — Problem and Goal
- Goal: unified platform for biological microscopy data storage and processing.
- Domain: RNAi screening against microsporidiosis in bees/silkworm-related assays.

## Slide 2 — Relevance
- Manual counting is slow and error-prone.
- Need reproducible storage + quick analytical feedback.

## Slide 3 — Database Design
- Show ER diagram (`docs/schemas/er_diagram.puml` export).
- Explain why `Project -> Probe -> Sample -> Image` hierarchy matches experiment workflow.
- Mention FK and cascade delete guarantees.

## Slide 4 — Stack Choice
- FastAPI + SQLAlchemy + PostgreSQL + React + Plotly + Docker.
- Why: Python ecosystem reuse, API productivity, reproducibility.

## Slide 5 — Backend Components
- CRUD API per entity.
- Upload handler.
- Analysis service that reuses current `bees` pipeline.

## Slide 6 — Frontend Components
- CRUD tables/forms.
- Edit + delete actions.
- Analyze button and status flow.

## Slide 7 — Analytics Output
- Results table: `Species | Probe | Mean | Std | n | P-value`.
- Chart on dashboard (mean titer by probe).

## Slide 8 — Live Demo Script
- Create project.
- Add species/probe/sample.
- Upload images.
- Run analysis.
- Show result table + graph.
- Delete one record to prove full CRUD.

## Slide 9 — Testing and Reliability
- Pytest API tests.
- CI run for lint/tests.
- Docker one-command start.

## Slide 10 — Results and Next Steps
- Summarize achieved requirements.
- Roadmap to push accuracy towards 90% with tuned YOLO/SAHI.
