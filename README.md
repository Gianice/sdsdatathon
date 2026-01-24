# SDS Datathon â€“ Champions Cluster Intelligence

## Project Overview

This project prepares and clusters company data, then uses a Gemini LLM to generate a humanâ€‘readable â€œcluster intelligenceâ€ report. The notebook (`notebooks/final_result.ipynb`) performs data prep and clustering, and the LLM script (`notebooks/llm.py`) summarizes each cluster into commercial insights and writes a Markdown report.

## Prerequisites

- **Python** 3.10+ (tested with 3.13.6).
- **pip** and a virtual environment tool (`venv` or `conda`).
- **Gemini API key** in `GEMINI_API_KEY` (used by `google-genai`).
- **Input data**: `raw_data/champions_group_data_with_cluster.csv` (clustered dataset).

## Step-by-Step Setup Guide

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your Gemini API key (recommended via `.env` at repo root):

   ```bash
   # .env
   GEMINI_API_KEY=your_key_here
   ```

## How to Run

There are two parts to this workflow. You must run the Notebook first to generate the clusters, followed by the LLM script to generate the report.

### Step 1: Run the Clustering Analysis

The notebook performs the heavy lifting for data processing and visualization.

Start the Jupyter server:

**Bash**
```bash
jupyter notebook
```

In the browser window that opens, navigate to notebooks/final_result.ipynb.

Click Run > Run All Cells.

This will process the raw data and save the intermediate clustered file (e.g., champions_group_data_with_cluster.csv).

### Step 2: Generate the LLM Report

Once the notebook has finished and the data is ready, run the Python script to trigger the Gemini LLM.

In your terminal (with the virtual environment still active), run:

**Bash**
```bash
python notebooks/llm.py
```

### Results

After the script finishes, you will find the generated report here:

Report Location: notebooks/champions_cluster_intelligence.md
