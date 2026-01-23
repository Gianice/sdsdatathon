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

# Use stable model to avoid 404s
MODEL_ID = "gemini-2.5-flash"

BATCH_SIZE = 10 
TOP_K_PER_SIDE_FOR_LLM = 5 
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.2
MAX_RETRIES = 6

# ----------------------------
# 1. Data Processing Helpers
# ----------------------------

def make_cluster_cards(profiles_df: pd.DataFrame, top_k: int = 8) -> list[dict]:
    """Identify what makes each cluster statistically unique."""
    global_mean = profiles_df.mean(axis=0)
    diff = profiles_df.subtract(global_mean, axis=1)

    cards = []
    for cid in profiles_df.index:
        d = diff.loc[cid].sort_values(ascending=False)
        top_pos = d.head(top_k)
        
        cards.append({
            "cluster_id": int(cid),
            "top_positive_features": [{"feature": f, "delta_from_typical": float(v)} for f, v in top_pos.items()],
        })
    return cards

def compress_cards_for_llm(cards: list[dict], top_k: int = 6) -> list[dict]:
    """Simplify the payload for the LLM."""
    small = []
    for c in cards:
        small.append({
            "cluster_id": c["cluster_id"],
            "top_positive_features": [
                {"feature": x["feature"], "delta_from_typical": round(float(x["delta_from_typical"]), 3)}
                for x in c["top_positive_features"][:top_k]
            ]
        })
    return small

def _nonnull_rate(series: pd.Series) -> float:
    if series is None or len(series) == 0: return 0.0
    return round(float(series.notna().mean()), 3)

def _top_share(series: pd.Series, k: int = 5) -> list[str]:
    """Returns simple list of top values to save tokens."""
    if series is None or len(series) == 0: return []
    return list(series.fillna("Unknown").astype(str).value_counts().head(k).index)

def make_cluster_context(raw_with_cluster: pd.DataFrame, cluster_col: str, cid: int) -> dict:
    """
    Extracts the 'Commercial Signals' from the raw data.
    """
    sub = raw_with_cluster[raw_with_cluster[cluster_col] == cid].copy()
    n = int(len(sub))
    
    # Helper to get numeric stats
    def _get_median(col):
        if col not in sub.columns: return 0
        s = pd.to_numeric(sub[col], errors="coerce")
        return float(s.median()) if not s.empty else 0

    # Extract Key Business Columns
    # Note: Adjust column names if your CSV is slightly different
    it_budget = sub["IT Budget"] if "IT Budget" in sub.columns else None
    servers = sub["No. of Servers"] if "No. of Servers" in sub.columns else None
    website = sub["Website"] if "Website" in sub.columns else None
    entity_type = sub["Entity Type"] if "Entity Type" in sub.columns else None
    
    # Calculate "Tech Intensity" proxies
    has_servers = _nonnull_rate(servers)
    has_it_budget = _nonnull_rate(it_budget)
    
    return {
        "n_records": n,
        "top_locations": _top_share(sub.get("Country"), 2),
        "top_industries": _top_share(sub.get("SIC Description"), 2),
        "top_entity_types": _top_share(entity_type, 2),
        "median_revenue": _get_median("Revenue (USD)"),
        "median_employees": _get_median("Employees Total"),
        "signals": {
            "has_website_rate": _nonnull_rate(website),
            "tech_data_availability": (has_servers + has_it_budget) / 2 # Simple score
        }
    }

def attach_context_to_cards(cards: list[dict], raw_df: pd.DataFrame, cluster_col: str) -> list[dict]:
    out = []
    for c in cards:
        c2 = dict(c)
        c2["cluster_context"] = make_cluster_context(raw_df, cluster_col, int(c["cluster_id"]))
        out.append(c2)
    return out

def make_lightweight_summary(cards_with_context: list[dict]) -> list[dict]:
    """
    Summarizes all clusters for the 'Executive Landscape' view.
    """
    summary = []
    for c in cards_with_context:
        ctx = c["cluster_context"]
        
        # Safe string formatting
        loc = ctx['top_locations'][0] if ctx['top_locations'] else "Unknown"
        ind = ctx['top_industries'][0] if ctx['top_industries'] else "Unknown"
        
        summary.append({
            "id": c["cluster_id"],
            "count": ctx["n_records"],
            "primary_market": f"{loc} - {ind}",
            "rev_median": ctx["median_revenue"],
            "tech_score": int(ctx["signals"]["tech_data_availability"] * 100),
            "website_pct": int(ctx["signals"]["has_website_rate"] * 100)
        })
    return summary

# ----------------------------
# 2. Strategic Prompts
# ----------------------------

def build_profile_prompt(batch_cards: list[dict]) -> str:
    """
    Asks for a Sales-Focused Profile per cluster.
    """
    return f"""
### Role
You are a B2B Market Analyst.

### Task
Analyze these company clusters. For EACH cluster, write a "Sales Intelligence Card".

### Input Data
{json.dumps(batch_cards, indent=2)}

### Output Format (Repeat for each cluster)

#### 📂 Cluster [ID]: [Professional Persona Name]
* **The Profile:** [Count] companies primarily in [Industry] located in [Location].
* **Commercial Signals:**
    * *Scale:* Median Revenue $[Rev] | Median Employees [Emp]
    * *Tech Signal:* (Interpret `tech_data_availability` - Do we have IT budget/server data?)
    * *Digital Footprint:* (Interpret `has_website_rate` - High or Low?)
* **Key Insight:** (Look at `top_positive_features` - e.g., "These companies are unusually likely to be manufacturing branches").
* **🎯 Target For:** (Who should buy this list? e.g., "Logistics providers," "IT Hardware sellers," "Risk Assessors").
---
""".strip()

def build_landscape_prompt(all_summaries: list[dict]) -> str:
    """
    Asks for the 'Superlatives' and Risk/Value analysis.
    """
    return f"""
### Role
You are a Data Monetization Strategist. Your client is selling a B2B Company Dataset.
Your goal is to demonstrate the value of this data by highlighting the most valuable, risky, and unique segments.

### Input: Cluster Summaries
{json.dumps(all_summaries, indent=2)}

### Task: Generate an "Executive Portfolio Review"

#### Part 1: The "Superlatives" (Pick specific Clusters)
Identify the specific Cluster ID that best fits each category below and explain WHY based on the data.
* **🏆 The "Whales" (High Value Target):** Which cluster has the highest median revenue/scale? (Best for Enterprise Sales).
* **💻 The "Tech-Ready" (Digital Target):** Which cluster has the highest `tech_score`? (Best for selling Software/Hardware).
* **⚠️ The "Dark Market" (High Risk):** Which cluster has high revenue but very low `website_pct` (no website)? These are often traditional, opaque, or risky firms.
* **💎 The "Volume Play" (SMB Target):** Which cluster has a high count of companies but smaller employee size? (Best for mass-market SaaS).

#### Part 2: Strategic Landscape Table
Create a markdown table summarizing the top 5 largest clusters:
| ID | Persona Name | Primary Market | Avg. Revenue | Best Product to Sell to Them |
|:---|:---|:---|:---|:---|
| ... | ... | ... | ... | ... |

#### Part 3: Commercial Pitch
Write one sentence to a potential buyer: "This dataset allows you to distinguish between [Cluster X Persona] and [Cluster Y Persona], enabling you to target [Specific Need]."
""".strip()

# ----------------------------
# 3. Robust Execution
# ----------------------------

def call_gemini_with_retries(client: genai.Client, prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config={"temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS},
            )
            # CHECK: Ensure response is valid text
            if response.text:
                return response.text
            else:
                return "\n⚠️ [System] Cluster batch skipped due to Safety Filter.\n"
                
        except Exception as e:
            # Handle Rate Limits
            err = str(e).lower()
            if "429" in err or "quota" in err:
                sleep_time = 30 + (attempt * 10)
                print(f"[Warning] Rate Limit Hit. Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
            elif "500" in err or "503" in err:
                print(f"[Retry {attempt+1}] Server Error. Retrying...")
                time.sleep(5)
            else:
                print(f"[Retry {attempt+1}] Unexpected Error: {e}")
                time.sleep(2)
    
    # FALLBACK: Return a string error message instead of None to prevent crash
    return "\n\n### ⚠️ Error: Could not generate this section after multiple attempts.\n"

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY in your environment.")
    
    client = genai.Client(api_key=api_key)

    # 1. Load Data
    print("Loading data...")
    if not PROFILES_PATH.exists() or not CLUSTERED_DATA_PATH.exists():
        print(f"Error: Files not found.\nExpected Profiles: {PROFILES_PATH}\nExpected Raw: {CLUSTERED_DATA_PATH}")
        return

    profiles = pd.read_csv(PROFILES_PATH, index_col=0).T
    profiles.index = profiles.index.astype(int)
    # FIX: low_memory=False prevents DtypeWarning on large files
    raw_with_cluster = pd.read_csv(CLUSTERED_DATA_PATH, low_memory=False)

    # 2. Prepare Data
    print("Generating cluster cards...")
    cluster_cards = make_cluster_cards(profiles, top_k=8)
    cards_for_llm = compress_cards_for_llm(cluster_cards, top_k=TOP_K_PER_SIDE_FOR_LLM)
    cards_with_context = attach_context_to_cards(cards_for_llm, raw_with_cluster, CLUSTER_COL)
    
    total_clusters = len(cards_with_context)
    print(f"Processing {total_clusters} clusters...")

    # ---------------------------------------------------------
    # PASS 1: Generate Landscape (The "Big Picture")
    # ---------------------------------------------------------
    print("\n--- Generating Executive Landscape (Pass 1) ---")
    all_summaries = make_lightweight_summary(cards_with_context)
    landscape_prompt = build_landscape_prompt(all_summaries)
    landscape_text = call_gemini_with_retries(client, landscape_prompt)
    print("Landscape analysis complete.")
    
    #  - Visualizing clusters by Risk vs Reward can help here.

    # ---------------------------------------------------------
    # PASS 2: Detailed Profiles (The "Deep Dive")
    # ---------------------------------------------------------
    print("\n--- Generating Detailed Profiles (Pass 2) ---")
    full_report_parts = []
    
    for i in range(0, total_clusters, BATCH_SIZE):
        batch = cards_with_context[i : i + BATCH_SIZE]
        print(f"  > Processing Batch {i}-{min(i+BATCH_SIZE, total_clusters)}")
        
        prompt = build_profile_prompt(batch)
        response_text = call_gemini_with_retries(client, prompt)
        full_report_parts.append(response_text)
        
        # Polite sleep to respect Free Tier
        time.sleep(2)

    # ---------------------------------------------------------
    # Final Output Assembly
    # ---------------------------------------------------------
    final_output = "# CHAMPIONS GROUP: COMMERCIAL INTELLIGENCE REPORT\n\n"
    
    # Add Landscape (Check if valid)
    if landscape_text and "Error" not in landscape_text:
        final_output += landscape_text + "\n\n"
    
    final_output += "## CLUSTER INTELLIGENCE CARDS\n"
    
    # FIX: Filter out None values before joining
    valid_parts = [part for part in full_report_parts if part is not None]
    final_output += "\n".join(valid_parts)

    out_path = Path(__file__).resolve().parent / "champions_cluster_insight.md"
    out_path.write_text(final_output, encoding="utf-8")
    
    print(f"\nSUCCESS! Report saved to: {out_path}")

if __name__ == "__main__":
    main()