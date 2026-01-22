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

# Your cluster profile CSV (rows=features, columns=cluster_ids) -> you transpose it back
PROFILES_PATH = Path(__file__).resolve().parent / "../data/processed/cluster_profiles_no_noise_transposed.csv"


# It should include the original columns like Country/City/State/SIC/etc.
RAW_DATA_PATH = Path(__file__).resolve().parent / "../raw_data/champions_group_data.csv"

# If you already have raw + cluster_id in one file, you can point RAW_DATA_PATH to that
# and set CLUSTERED_DATA_PATH = RAW_DATA_PATH.
CLUSTERED_DATA_PATH = Path(__file__).resolve().parent / "../raw_data/champions_group_data_with_cluster.csv"

# The column in CLUSTERED_DATA_PATH that stores the cluster label
CLUSTER_COL = "cluster_id"

MODEL_ID = "gemini-2.5-flash"

N_CLUSTERS_TO_SUMMARIZE = 5
TOP_K_PER_SIDE_FOR_LLM = 6

MAX_OUTPUT_TOKENS = 16000
TEMPERATURE = 0.2
MAX_RETRIES = 6


# ----------------------------
# Helpers: cluster cards (from profile deltas)
# ----------------------------

def make_cluster_cards(profiles_df: pd.DataFrame, top_k: int = 8) -> list[dict]:
    """
    Build cluster cards describing what is unusually high/low per cluster.
    profiles_df: rows=cluster_id, cols=features
    """
    global_mean = profiles_df.mean(axis=0)
    diff = profiles_df.subtract(global_mean, axis=1)

    cards = []
    for cid in profiles_df.index:
        d = diff.loc[cid].sort_values(ascending=False)
        top_pos = d.head(top_k)
        top_neg = d.tail(top_k)

        cards.append(
            {
                "cluster_id": int(cid),
                "top_positive_features": [
                    {"feature": f, "delta_from_typical": float(v)} for f, v in top_pos.items()
                ],
                "top_negative_features": [
                    {"feature": f, "delta_from_typical": float(v)} for f, v in top_neg.items()
                ],
            }
        )
    return cards


def compress_cards_for_llm(cards: list[dict], top_k: int = 6) -> list[dict]:
    """
    Reduce payload size so the request is faster and less likely to time out.
    """
    small = []
    for c in cards:
        small.append(
            {
                "cluster_id": c["cluster_id"],
                "top_positive_features": [
                    {"feature": x["feature"], "delta_from_typical": round(float(x["delta_from_typical"]), 3)}
                    for x in c["top_positive_features"][:top_k]
                ],
                "top_negative_features": [
                    {"feature": x["feature"], "delta_from_typical": round(float(x["delta_from_typical"]), 3)}
                    for x in c["top_negative_features"][:top_k]
                ],
            }
        )
    return small


# ----------------------------
# Helpers: real-world context (from raw fields)
# ----------------------------

def _top_share(series: pd.Series, k: int = 5) -> list[dict[str, Any]]:
    """Top-k value distribution as share of rows."""
    if series is None or len(series) == 0:
        return []
    s = series.fillna("NA").astype(str).str.strip()
    vc = s.value_counts(dropna=False).head(k)
    n = int(vc.sum())
    out = []
    for val, cnt in vc.items():
        out.append({"value": val, "share": round(float(cnt) / float(len(s)), 3), "count": int(cnt)})
    return out


def _nonnull_rate(series: pd.Series) -> float | None:
    if series is None or len(series) == 0:
        return None
    return round(float(series.notna().mean()), 3)


def _maybe_col(df: pd.DataFrame, col: str) -> pd.Series | None:
    return df[col] if col in df.columns else None


def make_cluster_context(raw_with_cluster: pd.DataFrame, cluster_col: str, cid: int) -> dict:
    """
    Compute a compact, business-readable summary of a cluster from raw columns.
    Keep this small, because it will be sent to the LLM.
    """
    sub = raw_with_cluster[raw_with_cluster[cluster_col] == cid].copy()
    n = int(len(sub))

    # Basic numeric summaries (only if present)
    def _num_summary(col: str) -> dict | None:
        if col not in sub.columns or n == 0:
            return None
        s = pd.to_numeric(sub[col], errors="coerce")
        if s.notna().sum() == 0:
            return None
        return {
            "median": float(s.median()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "non_null_rate": float(s.notna().mean()),
        }

    # Common columns in your raw schema
    country = _maybe_col(sub, "Country")
    region = _maybe_col(sub, "Region")
    state = _maybe_col(sub, "State")
    city = _maybe_col(sub, "City")

    entity_type = _maybe_col(sub, "Entity Type")
    ownership = _maybe_col(sub, "Ownership Type")

    sic_desc = _maybe_col(sub, "SIC Description")
    naics_desc = _maybe_col(sub, "NAICS Description")
    isic_desc = _maybe_col(sub, "ISIC Rev 4 Description")
    anzsic_desc = _maybe_col(sub, "ANZSIC Description")

    website = _maybe_col(sub, "Website")
    phone = _maybe_col(sub, "Phone Number")
    company_desc = _maybe_col(sub, "Company Description")
    reg_no = _maybe_col(sub, "Registration Number")

    # Your raw dataset seems to store IT as ranges ("1 to 10") rather than numeric
    # We'll summarize missingness + top buckets for a few
    it_pc = _maybe_col(sub, "No. of PC")
    it_servers = _maybe_col(sub, "No. of Servers")
    it_laptops = _maybe_col(sub, "No. of Laptops")
    it_desktops = _maybe_col(sub, "No. of Desktops")
    it_budget = _maybe_col(sub, "IT Budget")
    it_spend = _maybe_col(sub, "IT spend")

    ctx = {
        "n_records": n,
        "top_countries": _top_share(country, 3),
        "top_regions": _top_share(region, 3),
        "top_states": _top_share(state, 5),
        "top_cities": _top_share(city, 5),
        "entity_type_dist": _top_share(entity_type, 5),
        "ownership_type_dist": _top_share(ownership, 5),
        "top_industries_sic": _top_share(sic_desc, 5),
        "top_industries_naics": _top_share(naics_desc, 5),
        "top_industries_isic": _top_share(isic_desc, 5),
        "top_industries_anzsic": _top_share(anzsic_desc, 5),
        "data_availability_rates": {
            "has_website_rate": _nonnull_rate(website),
            "has_phone_rate": _nonnull_rate(phone),
            "has_company_description_rate": _nonnull_rate(company_desc),
            "has_registration_number_rate": _nonnull_rate(reg_no),
        },
        "it_field_availability_rates": {
            "pc_non_null_rate": _nonnull_rate(it_pc),
            "servers_non_null_rate": _nonnull_rate(it_servers),
            "laptops_non_null_rate": _nonnull_rate(it_laptops),
            "desktops_non_null_rate": _nonnull_rate(it_desktops),
            "it_budget_non_null_rate": _nonnull_rate(it_budget),
            "it_spend_non_null_rate": _nonnull_rate(it_spend),
        },
        "top_it_pc_buckets": _top_share(it_pc, 5),
        "top_it_servers_buckets": _top_share(it_servers, 5),
    }

    # Optional numeric summaries (if your columns are numeric)
    # If these are strings or missing, they'll be None.
    ctx["employees_total_summary"] = _num_summary("Employees Total")
    ctx["revenue_usd_summary"] = _num_summary("Revenue (USD)")
    ctx["year_founded_summary"] = _num_summary("Year Found")

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
# Prompt builder (UPDATED to use cluster_context)
# ----------------------------

def build_prompt(cards: list[dict], extra_context: str = "") -> str:
    return f"""

Extra context from user: {extra_context}

### Role
You are an expert Business Intelligence Analyst specializing in firmographic segmentation. Your goal is to transform raw cluster statistics into "Company Intelligence" reports that help strategic decision-makers understand different market segments.

### Input Data
Below is a JSON list of "Cluster Cards." Each card represents a distinct group of companies identified through machine learning.
{json.dumps(cards, indent=2)}

### Instructions
For each cluster provided in the JSON, generate a structured insight report covering the following:

1. **The Cluster Persona**: Create a professional, descriptive name for this group (e.g., "Digital-Native Asian Manufacturers" or "Established Western Service Firms"). Provide a 2-line executive summary of who these companies are., - **Scale**: Explicitly state: "This cluster contains **[n_records]** companies."

2. **Defining Characteristics**: Analyze the `top_positive_features` and `top_negative_features`. Explain in plain English what makes this group unique compared to the "typical" company in the dataset.

3. **Geographic & Operational Footprint**: 
   - Synthesize the `top_countries`, `top_states`, and `top_cities`. 
   - Comment on the `data_availability_rates`. (e.g., If `has_website_rate` is low, what does that imply about their digital presence or the nature of their business?)

4. **Financial & Scale Profile**: Use the `employees_total_summary` and `revenue_usd_summary`. Describe the scale of these firms (Small/Medium/Large) and their economic maturity using the `year_founded_summary`.

5. **Technology Maturity**: Interpret the `it_field_availability_rates` and `top_it_pc_buckets`. Are these technology-heavy operations or traditional low-tech firms?

6. **Commercial "So-What?"**: Provide one specific actionable insight for a:
   - **Sales Leader**: How should they pitch to this cluster?
   - **Risk Manager**: What is the primary risk of dealing with this group?

#### Cross-Cluster Comparison & Landscape Analysis
After profiling the individual clusters, provide a "Market Comparison" section:

1.  **The "High vs. Low" Matrix**: Identify which clusters are the "Leaders" in technology (using `it_field_availability_rates`) versus those that are "Traditional."
2.  **Growth vs. Stability**: Compare the `year_founded_summary` across clusters. Which group represents the "New Wave" vs. the "Established Core"?
3.  **Geographic Moats**: Highlight if certain clusters are hyper-localized to one region (e.g., China/Guangdong) compared to clusters with a broader global footprint.
4.  **The "Hidden Gem"**: Identify which cluster has the highest `data_availability_rates` but smaller company sizes—often a sign of a high-quality, underserved market.

### Output Format
- Use Markdown for clear hierarchy.
- Use bullet points for readability.
- Avoid mentioning "Cluster ID" as the primary title; use the "Persona Name" instead.
- **Strict Rule**: Do not simply list the JSON values. Synthesize them into observations.


==========================
CLUSTER CARDS (INPUT JSON)
==========================
{json.dumps(cards, indent=2)}
""".strip()


# ----------------------------
# Gemini call with retries
# ----------------------------

def call_gemini_with_retries(client: genai.Client, prompt: str) -> str:
    last_err = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
            )
            return response.text

        except Exception as e:
            last_err = e
            msg = str(e).lower()

            transient = (
                "429" in msg
                or "rate" in msg
                or "quota" in msg
                or "timeout" in msg
                or "503" in msg
                or "500" in msg
                or "unavailable" in msg
            )

            if not transient and attempt >= 1:
                raise

            if attempt == MAX_RETRIES - 1:
                break

            sleep_s = (2 ** attempt) + random.random()
            print(f"[Retry {attempt+1}/{MAX_RETRIES}] Temporary Gemini error. Sleeping {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} retries: {last_err}") from last_err


# ----------------------------
# Main
# ----------------------------

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Gemini API key not found. Put GEMINI_API_KEY=... in your .env file.")

    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Profiles CSV not found at: {PROFILES_PATH}")

    if not CLUSTERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clustered raw data not found at: {CLUSTERED_DATA_PATH}\n"
            f"Expected a file that contains raw columns + a '{CLUSTER_COL}' column."
        )

    # Load profiles (cluster profiles)
    profiles = pd.read_csv(PROFILES_PATH, index_col=0).T
    profiles.index = profiles.index.astype(int)
    print("Loaded profiles:", profiles.shape)

    # Load raw data WITH cluster labels
    raw_with_cluster = pd.read_csv(CLUSTERED_DATA_PATH)
    if CLUSTER_COL not in raw_with_cluster.columns:
        raise ValueError(
            f"'{CLUSTER_COL}' not found in {CLUSTERED_DATA_PATH}.\n"
            f"Please ensure your raw dataset includes a column named '{CLUSTER_COL}'."
        )

    # Make delta cards
    cluster_cards = make_cluster_cards(profiles, top_k=8)
    cards_for_llm = compress_cards_for_llm(
        cluster_cards[:N_CLUSTERS_TO_SUMMARIZE],
        top_k=TOP_K_PER_SIDE_FOR_LLM
    )

    # Attach real-world context from raw columns
    cards_with_context = attach_context_to_cards(cards_for_llm, raw_with_cluster, CLUSTER_COL)

    # Build prompt
    prompt = build_prompt(
        cards_with_context,
        extra_context="The goal is to prove commercial value of dataset and show within + cross cluster differences.",
    )

    # Call Gemini
    client = genai.Client(api_key=api_key)
    final_report = call_gemini_with_retries(client, prompt)

    # Save output
    out_path = Path(__file__).resolve().parent / "cluster_analysis.txt"
    out_path.write_text(final_report, encoding="utf-8")
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
