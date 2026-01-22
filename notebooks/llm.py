import os
import json
import time
import random
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google import genai

# ----------------------------
# Configuration
# ----------------------------
PROFILES_PATH = Path(__file__).resolve().parent / "../data/processed/cluster_profiles_no_noise_transposed.csv"
CLUSTERED_DATA_PATH = Path(__file__).resolve().parent / "../raw_data/champions_group_data_with_cluster.csv"
CLUSTER_COL = "cluster_id"

# FIXED: Use the stable 1.5 Flash model (2.5-flash-lite does not exist yet)
MODEL_ID = "gemini-3-flash-preview"

BATCH_SIZE = 10  # Increased batch size since outputs are now shorter
TOP_K_PER_SIDE_FOR_LLM = 5 
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.2
MAX_RETRIES = 6

# ----------------------------
# Helpers: Data Preparation (Kept same as before)
# ----------------------------

def make_cluster_cards(profiles_df: pd.DataFrame, top_k: int = 8) -> list[dict]:
    global_mean = profiles_df.mean(axis=0)
    diff = profiles_df.subtract(global_mean, axis=1)
    cards = []
    for cid in profiles_df.index:
        d = diff.loc[cid].sort_values(ascending=False)
        top_pos = d.head(top_k)
        top_neg = d.tail(top_k)
        cards.append({
            "cluster_id": int(cid),
            "top_positive_features": [{"feature": f, "delta_from_typical": float(v)} for f, v in top_pos.items()],
            "top_negative_features": [{"feature": f, "delta_from_typical": float(v)} for f, v in top_neg.items()],
        })
    return cards

def compress_cards_for_llm(cards: list[dict], top_k: int = 6) -> list[dict]:
    small = []
    for c in cards:
        small.append({
            "cluster_id": c["cluster_id"],
            "top_positive_features": [
                {"feature": x["feature"], "delta_from_typical": round(float(x["delta_from_typical"]), 3)}
                for x in c["top_positive_features"][:top_k]
            ],
            # We remove negative features for the "short" version to save tokens
            # unless truly needed. Here we keep them but limit count.
             "top_negative_features": [
                {"feature": x["feature"], "delta_from_typical": round(float(x["delta_from_typical"]), 3)}
                for x in c["top_negative_features"][:2] 
            ],
        })
    return small

def make_lightweight_summary(cards_with_context: list[dict]) -> list[dict]:
    summary = []
    for c in cards_with_context:
        ctx = c["cluster_context"]
        top_ind = ctx["top_industries_sic"][0]["value"] if ctx["top_industries_sic"] else "Unknown"
        top_country = ctx["top_countries"][0]["value"] if ctx["top_countries"] else "Unknown"
        summary.append({
            "id": c["cluster_id"],
            "count": ctx["n_records"],
            "top_industry": top_ind,
            "top_country": top_country,
            "median_revenue": ctx.get("revenue_usd_summary", {}).get("median"),
            "website_rate": ctx["data_availability_rates"]["has_website_rate"],
        })
    return summary

# ----------------------------
# Helpers: Context (Kept same as before)
# ----------------------------

def _top_share(series: pd.Series, k: int = 5) -> list[dict[str, Any]]:
    if series is None or len(series) == 0: return []
    s = series.fillna("NA").astype(str).str.strip()
    vc = s.value_counts(dropna=False).head(k)
    out = []
    for val, cnt in vc.items():
        out.append({"value": val, "share": round(float(cnt) / float(len(s)), 3), "count": int(cnt)})
    return out

def _nonnull_rate(series: pd.Series) -> float | None:
    if series is None or len(series) == 0: return None
    return round(float(series.notna().mean()), 3)

def _maybe_col(df: pd.DataFrame, col: str) -> pd.Series | None:
    return df[col] if col in df.columns else None

def make_cluster_context(raw_with_cluster: pd.DataFrame, cluster_col: str, cid: int) -> dict:
    sub = raw_with_cluster[raw_with_cluster[cluster_col] == cid].copy()
    n = int(len(sub))

    def _num_summary(col: str) -> dict | None:
        if col not in sub.columns or n == 0: return None
        s = pd.to_numeric(sub[col], errors="coerce")
        if s.notna().sum() == 0: return None
        return { "median": float(s.median()) } # Keep only median for brevity

    country = _maybe_col(sub, "Country")
    sic_desc = _maybe_col(sub, "SIC Description")
    website = _maybe_col(sub, "Website")
    it_pc = _maybe_col(sub, "No. of PC")

    ctx = {
        "n_records": n,
        "top_countries": _top_share(country, 2), # Reduce to top 2
        "top_industries_sic": _top_share(sic_desc, 2),
        "data_availability_rates": { "has_website_rate": _nonnull_rate(website) },
        "it_field_availability_rates": { "pc_non_null_rate": _nonnull_rate(it_pc) },
    }
    ctx["revenue_usd_summary"] = _num_summary("Revenue (USD)")
    ctx["employees_total_summary"] = _num_summary("Employees Total")
    return ctx

def attach_context_to_cards(cards_for_llm: list[dict], raw_with_cluster: pd.DataFrame, cluster_col: str) -> list[dict]:
    out = []
    for c in cards_for_llm:
        cid = int(c["cluster_id"])
        c2 = dict(c)
        c2["cluster_context"] = make_cluster_context(raw_with_cluster, cluster_col, cid)
        out.append(c2)
    return out

# ----------------------------
# UPDATED: Concise Prompts
# ----------------------------

def build_profile_prompt(batch_cards: list[dict]) -> str:
    """
    Asks for a 'Trading Card' style output: minimal text, maximum signal.
    """
    return f"""
### Role
You are a Database Intelligence Specialist.

### Task
Create a concise "Segment Card" for each company cluster below. 
Do NOT write paragraphs. Use a strict bulleted format.

### Input Data
{json.dumps(batch_cards, indent=2)}

### Output Format (Repeat for each cluster)

#### Cluster [ID]: [Creative Professional Name]
* **Vital Stats:** [N] Companies | Top Loc: [Country] | Top Ind: [Industry]
* **The DNA:** 1 short sentence explaining what makes them unique based on `top_positive_features`.
* **Commercial Signal:** 1 short sentence on the sales opportunity (e.g., "High revenue but low tech—sell digital transformation").

---
""".strip()

def build_landscape_prompt(all_summaries: list[dict]) -> str:
    """
    Asks for a Comparative Table as the main output.
    """
    return f"""
### Role
You are a Strategy Director.

### Task
Create a **Market Landscape Summary** for the following clusters.
Your goal is to compare them in a single view.

### Input Data (Summary of All Clusters)
{json.dumps(all_summaries, indent=2)}

### Output Format

## 1. The Market at a Glance
(Provide 3 bullet points summarizing the dominant patterns in the dataset).

## 2. Cluster Comparison Matrix
(Create a Markdown Table with columns: Cluster ID, Persona Name (invent one), Count, Primary Industry, Digital Maturity (High/Low based on website_rate)).

## 3. Top Targets
(Identify the single best cluster for: 1. High Volume Sales, 2. High Value/Enterprise Sales).
""".strip()

# ----------------------------
# Gemini Client
# ----------------------------

def call_gemini_with_retries(client: genai.Client, prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config={"temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS},
            )
            # Check if text was actually generated (sometimes safety filters return empty)
            if response.text:
                return response.text
            else:
                return " [System] Cluster skipped due to Safety Filter."
                
        except Exception as e:
            # Handle rate limits
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(30 + (attempt * 10))
            else:
                time.sleep(2)
                
    # FALLBACK: Return a string error message instead of None
    return f"\n\n###  Error: Could not generate this section.\n"

# ----------------------------
# Main
# ----------------------------

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # 1. Load Data
    print("Loading data...")
    profiles = pd.read_csv(PROFILES_PATH, index_col=0).T
    profiles.index = profiles.index.astype(int)
    raw_with_cluster = pd.read_csv(CLUSTERED_DATA_PATH)

    # 2. Prepare Data
    print("Generating cluster cards...")
    cluster_cards = make_cluster_cards(profiles, top_k=8)
    cards_for_llm = compress_cards_for_llm(cluster_cards, top_k=TOP_K_PER_SIDE_FOR_LLM)
    cards_with_context = attach_context_to_cards(cards_for_llm, raw_with_cluster, CLUSTER_COL)
    
    total_clusters = len(cards_with_context)
    print(f"Processing {total_clusters} clusters...")

    # ---------------------------------------------------------
    # PASS 1: Detailed Profiling (Short & Clean)
    # ---------------------------------------------------------
    full_report_parts = []
    print(" Generating Profiles...")
    
    for i in range(0, total_clusters, BATCH_SIZE):
        batch = cards_with_context[i : i + BATCH_SIZE]
        print(f"  > Batch {i}-{i+len(batch)}")
        
        prompt = build_profile_prompt(batch)
        response_text = call_gemini_with_retries(client, prompt)
        full_report_parts.append(response_text)
        
        time.sleep(1)

    # ---------------------------------------------------------
    # PASS 2: Landscape Analysis (The Cross-Cluster Part)
    # ---------------------------------------------------------
    print(" Generating Landscape Analysis...")
    all_cluster_summaries = make_lightweight_summary(cards_with_context)
    landscape_prompt = build_landscape_prompt(all_cluster_summaries)
    landscape_text = call_gemini_with_retries(client, landscape_prompt)

    # ---------------------------------------------------------
    # Final Output
    # ---------------------------------------------------------
    final_output = "# CHAMPIONS GROUP: MARKET INTELLIGENCE\n\n"
    final_output += landscape_text + "\n\n"
    final_output += "## SEGMENT PROFILES\n"
    final_output += "\n".join(full_report_parts)

    out_path = Path(__file__).resolve().parent / "cluster_insights.md"
    out_path.write_text(final_output, encoding="utf-8")
    print(f"\nDone! Report saved to: {out_path}")

if __name__ == "__main__":
    main()