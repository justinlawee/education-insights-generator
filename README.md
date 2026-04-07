# Education Insights Generator (Kellogg AI Lab × Innovare)

### 🧠 Turning Educational Data Into Actionable Insights

This repository contains the full **AI Insight Generation** and **Evaluation** system developed during the **Kellogg AI Lab × Innovare** collaboration. It uses **LLMs, multi-agent workflows, and educational data pipelines** to automatically generate, visualize, and evaluate insights from K–12 assessment datasets.

**Demo Video Included Below**

---

## 🚀 Project Overview

**Goal:**  
Empower educators and school leaders to interpret large-scale student performance data — turning spreadsheets into narratives and guiding questions using AI.

**Solution:**  
An end-to-end AI pipeline that connects **BigQuery** and **LangChain**, orchestrated through a multi-agent system (n8n + Lovable).  
The system performs:

1. **Data Ingestion** – pulls K–12 assessment data (e.g., IAR Math & ELA) from BigQuery.  
2. **Insight Generation** – uses prompt-driven LLMs to surface key findings and next-step questions.  
3. **Visualization** – builds interactive charts and summaries for district, school, and subgroup performance.  
4. **Evaluation (Judge LLM)** – scores each generated insight for clarity, actionability, and accuracy.  
5. **Feedback Loop** – integrates structured rubric scores back into the workflow to improve future outputs.

---

## 🧩 Architecture
```
BigQuery ─▶ ETL / Data Schema ─▶ LLM Insight Generator (LangChain) ─▶
Visualization (Streamlit / Plotly) ─▶ Judge LLM Evaluation ─▶ Feedback & Storage


### Core Components

| Layer | Description |
|-------|--------------|
| **Data Layer** | BigQuery tables for K–12 assessment (IAR Math & ELA). |
| **Processing Layer** | Python utilities for data cleaning and schema alignment (`src/bq_functions.py`, `src/utils.py`). |
| **AI Layer** | Insight generation via prompt templates and LLM orchestration (`src/prompts.py`, `src/workflow.py`). |
| **Visualization Layer** | Dashboards and report generation using Streamlit & Plotly (`innovare_dashboard.ipynb`). |
| **Evaluation Layer** | “Judge LLM” that scores insights using a rubric for quality and bias detection. |
| **Automation Layer** | Multi-agent workflow designed in **n8n**, with optional **Lovable** front-end for nontechnical users. |
```
---

## 📂 Repository Structure

```
education-insights-generator/
│
├── src/ # Core logic and modular utilities
│ ├── workflow.py
│ ├── utils.py
│ ├── settings.py
│ ├── prompts.py
│ ├── bq_functions.py
│ └── __init__.py
│
├── notebooks/ # Jupyter exploration & insight notebooks
│ ├── CleanVersion_QGenerator.ipynb
│ ├── Insight_Generation_Consolidation.ipynb
│ ├── VisGenerator.ipynb
│ ├── innovare_dashboard.ipynb
│ ├── Judge_LLM.ipynb - Colab.pdf
│
├── data_samples/ # De-identified sample data
│ ├── tables_descriptions.csv
│ ├── iar_math_sample.csv
│ └── iar_ela_sample.csv
│
├── schema/ # Table schemas & metadata
│ └── schema_iar_ela.json
│ └── schema_iar_math.json
│ └── schema_table_descriptions.json
│
├── docs/ # Supporting documentation
│ ├── README_DATA.md
│ ├── README_GCLOUD.md
│ ├── README_TABLES.md
│ ├── Insights Matrix.pdf
│ └── innovare_n8n.json (private version omitted)
│
├── requirements.txt # Dependencies (LangChain, BigQuery, etc.)
├── Final Presentation Deck.pdf
├── Final Presentation Video.mov
└── README.md # This file
```

---

## ⚙️ Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Language / Frameworks** | Python · LangChain · Streamlit · Pandas · Plotly |
| **LLM Access** | Google Vertex AI · Gemini 1.5 Pro · Groq API |
| **Data & Infra** | BigQuery · GCP SDK · n8n Automation |
| **Evaluation** | Custom “Judge LLM” rubric system with prompt templating |
| **Visualization** | Lovable · Plotly Graphs |

---

## 🧮 Sample Workflow

```python
from src.workflow import generate_insights
from src.settings import PROJECT_ID, DATASET_ID

df = generate_insights(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    table="iar_math",
    llm_model="gemini-1.5-pro"
)
```

This function:
1. Queries BigQuery for assessment data.
2. Structures it into schema-aligned JSON.
3. Passes it through a prompt template.
4. Returns generated insights + AI-evaluated scores.

---

## 📊 Example Outputs

Example	Description
Insight:	“5th grade math proficiency grew 8 points year-over-year, driven by subgroup improvements in Hispanic and EL populations.”
Judge LLM Score:	4.6 / 5 — “Clear, actionable, and supported by data.”
Next Step Prompt:	“How might schools with >70% growth maintain gains next year?”

See Judge_LLM.ipynb for the scoring pipeline and examples.

## 🧠 Evaluation Framework (Judge LLM)

The **Judge LLM** evaluates insight quality across the following dimensions:

| Dimension | Description |
|------------|--------------|
| **Accuracy** | Alignment with underlying data |
| **Clarity** | Ease of interpretation by educators |
| **Actionability** | Whether the insight suggests next steps |
| **Bias Detection** | Flags overgeneralization or inequitable framing |

**Evaluation pipeline code lives in:**
- `notebooks/Judge_LLM.ipynb`
- `docs/Insights Matrix.pdf`

---

## 🔒 Privacy Notes

- All datasets in this repo are synthetic and de-identified.
- Real assessment data was stored in BigQuery and is not included here.
- innovare_n8n.json has been sanitized — API keys and internal endpoints removed.

---

📎 Additional Materials

- 🎥 [Demo Video (YouTube)](https://youtu.be/713CubB-iXM)
- 🧾 [Final Presentation (Google Drive)](https://drive.google.com/file/d/1KGBahx3mkoWIHvbtndkx1bl0wV25w8ij)
- ⚙️ [n8n Multi-Agent Flow (redacted JSON)](https://drive.google.com/file/d/1xUvA4_uastxhl175A02g-8PWJuA-n8Sc)

---

## ⭐️ Key Takeaways

Built a real applied AI system connecting LLMs, data pipelines, and human evaluation loops.
Serves as a blueprint for AI-assisted insight generation in any domain (education, analytics, or enterprise data).

---

## 🏫 About

Developed by:
Justin Borenstein-Lawee

In collaboration with:
Kellogg School of Management AI Lab (April-June, 2025)
