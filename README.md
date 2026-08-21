# 🚀 Downstream Activation Engine

> **Filtered Analytics & Executive Decision Support for Digital Marketing Campaigns**  
> *Translating top-of-funnel vanity traffic into true downstream business value.*

---

## 📌 Project Overview & Problem Statement

Digital marketing teams frequently track campaign metrics (impressions, click-through rates, signups) across disparate tools like Google Ads, Meta Ads, and Google Analytics 4[cite: 1, 2]. However, leadership often lacks visibility into which campaigns drive **meaningful downstream activation** versus high-volume **vanity traffic**[cite: 1, 2].

Allocating marketing budgets purely on Click-Through Rate (CTR) creates a strategic blind spot[cite: 1, 2]:
* **Vanity Traps:** High CTR campaigns frequently suffer from post-signup bounce rates.
* **Underfunded Drivers:** Low-volume, high-converting campaigns that yield genuine long-term active users are often defunded.

The **Downstream Activation Engine** bridges this gap by unifying cross-platform campaign logs with post-signup user events, evaluating campaigns based on **7-Day Activation Rates** and **Cost Per Activated User (CPAU)**[cite: 1, 2].

---

## ✨ Key Features

* **Multi-Source Data Ingestion & Validation:** Automatically ingests and validates schemas across impressions, clicks, signups, and user activity event logs[cite: 1, 2].
* **Entity Resolution & Multi-Key Joins:** Maps top-of-funnel campaign keys to post-signup user behavior logs while preventing record loss[cite: 1, 2].
* **Automated Activation Scoring:** Classifies users into *Vanity Traffic* vs. *Activated Users* using custom business logic (e.g., Signup + 1 key action within 7 days)[cite: 1, 2].
* **Interactive Executive Dashboard:** Built with Streamlit and Plotly to deliver real-time KPI scorecards, multi-stage conversion funnels, side-by-side CTR vs. Activation scatter plots, and budget reallocation flags[cite: 1, 2].
* **Automated Data Pipeline & Alerts:** Python execution script with logging and GitHub Actions automation for daily run checks and vanity spike warnings[cite: 1, 2].

---

## 🛠️ Tech Stack & Architecture

| Layer / Domain | Technologies & Frameworks | Purpose & Justification |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core language for script execution and data processing[cite: 1, 2]. |
| **Data Processing** | Pandas, NumPy | Vectorized transformations, null handling, string cleaning[cite: 1, 2]. |
| **Database & SQL** | SQLite / PostgreSQL, SQLAlchemy | Structured querying, window functions, aggregated views[cite: 1, 2]. |
| **Visualization & UI** | Streamlit, Plotly Express | Interactive dashboard, real-time filters, funnel charts[cite: 1, 2]. |
| **Automation & Quality** | PyYAML, Logging, GitHub Actions | Scheduled pipeline execution, validation scripts[cite: 1, 2]. |
| **Environment & Git** | Git, GitHub, `venv`, `requirements.txt` | Reproducible development workspace and team workflow[cite: 1, 2]. |

---

## 📂 Project Repository Structure

```text
Downstream-Activation-Engine/
├── .github/
│   └── workflows/
│       └── pipeline_validation.yml    # GitHub Actions workflow
├── data/
│   ├── raw/                            # Disparate marketing & user logs (CSV/JSON)
│   └── processed/                      # Cleaned, merged, and transformed master datasets
├── docs/
│   ├── Product_Requirement_Document_Sprint1.docx
│   └── data_dictionary.md              # Column descriptions & business mapping
├── src/
│   ├── __init__.py
│   ├── ingestion.py                    # Multi-source dataset intake & validation
│   ├── cleaning.py                     # Standardisation, text trimming & deduplication
│   ├── joining.py                      # Multi-source entity resolution & join rules
│   ├── feature_engineering.py          # 7-Day activation flags & derived columns
│   ├── sql_layer.py                    # SQL view creation & database aggregation
│   └── utils.py                        # Logging, YAML config loader, and helpers
├── app.py                              # Streamlit Executive Dashboard UI
├── pipeline_run.py                     # Single-run automated pipeline executor
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Ignored envs, cache files, and raw data outputs
└── README.md                           # Project documentation

.
