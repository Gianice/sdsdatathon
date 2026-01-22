import os
import json
import time
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai


# ----------------------------
# Configuration
# ----------------------------

PROFILES_PATH = Path(__file__).resolve().parent / "../data/processed/cluster_profiles_no_noise_transposed.csv"

MODEL_ID = "gemini-2.5-flash"

N_CLUSTERS_TO_SUMMARIZE = 5
TOP_K_PER_SIDE_FOR_LLM = 6

MAX_OUTPUT_TOKENS = 16000
TEMPERATURE = 0.2
MAX_RETRIES = 6


# ----------------------------
# Helpers
# ----------------------------

def make_cluster_cards(profiles_df: pd.DataFrame, top_k: int = 8) -> list[dict]:
    """
    Build cluster cards describing what is unusually high/low per cluster.
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



def build_prompt(cards: list[dict], extra_context: str = "") -> str:
    return f"""
You are a senior business analyst and data scientist interpreting HDBSCAN clustering results on company-level data.

## Dataset Context
- Data type: firmographic + contactability + registration fields
- Features: numeric (scaled) + one-hot encoded categorical flags
- Each cluster card includes:
  - cluster_id
  - size (may be null)
  - top positive features: higher than average (Δ > 0)
  - top negative features: lower than average (Δ < 0)

Extra context from user: {extra_context}

==========================
IMPORTANT INTERPRETATION RULES
==========================
1) Δ values are relative (higher/lower than average), NOT absolute real-world magnitude.
2) One-hot categories are mutually exclusive flags — interpret carefully.
3) Some features may represent missingness/unknown — treat as quality signals.
4) Small clusters may be unstable/noisy. If size is unknown, mention uncertainty.
5) Do NOT hallucinate fields that are not present in the card.
6) If feature names are unclear, interpret cautiously and label as "uncertain".
7) Write in business English, simple and direct, no excessive ML jargon.

==========================
YOUR OUTPUT FORMAT (STRICT)
==========================
Output MUST be Markdown only.
Do NOT output raw JSON.
Use the exact headings and structure below.

# Cluster Intelligence Report

## Part A — Deep Analysis Per Cluster
For EACH cluster, output the following:

### Cluster {{cluster_id}} — <Short Name (3–6 words)>
**1) Most Special Signal (1–2 sentences)**
- Identify the single most distinctive feature or pattern in this cluster (the strongest / most telling combination of high+low signals)
- Explain why it stands out compared to typical companies

**2) Cluster Identity (2–3 sentences)**
- Describe what type of companies are in this cluster using the feature signals
- Mention operational behavior / likely profile (e.g., size, region, maturity, data completeness)

**3) Key Feature Interpretation**
Use a two-column bullet style:
- ✅ Higher than average:
  - feature_name (Δ=...) → business meaning
  - feature_name (Δ=...) → business meaning
- ❌ Lower than average:
  - feature_name (Δ=...) → business meaning
  - feature_name (Δ=...) → business meaning

**4) Risk Signals (2–5 bullets)**
- Identify potential risks: fraud, incomplete registration, low contactability, inconsistent info, governance red flags, etc.
- If you detect missingness/unknown-related features, treat it as a data quality / operational risk.

**5) Commercial Value Signals (2–5 bullets)**
- Explain how this cluster can create value for potential buyers:
  - lead scoring / B2B targeting
  - onboarding checks
  - segmentation for pricing/marketing
  - compliance / risk screening
  - operational benchmarking

**6) Actionable Advice (3 bullets)**
Each bullet must be an action like:
- "Prioritize these companies for ____"
- "Flag these companies for ____"
- "Recommend additional verification of ____"

**7) Confidence (Low/Medium/High)**
1 short sentence explaining confidence based on:
- clarity of signals
- size stability (if unknown, say so)
- whether signals are coherent or contradictory

---

## Part B — Cross-Cluster Comparison & Insights (Very Important)
After all clusters are analyzed, write:

### 1) Cluster Map (Quick Grouping)
Group clusters into 2–5 higher-level families such as:
- “Highly reachable & well-registered”
- “Low data quality / risky”
- “Large established firms”
- “Small/fragmented or unusual profiles”
Use bullet lists.

### 2) Cross-Cluster Differences (5–10 bullets)
Compare clusters in terms of:
- risk profile
- contactability / data completeness
- business maturity signals
- sector/geography indicators (only if present)
- operational structure (HQ vs branches etc. if signals exist)

### 3) Risk Ranking (Top 3)
List the 3 clusters that look most risky and WHY.
If the ranking is uncertain, state that clearly.

### 4) Commercial Value Ranking (Top 3)
List the 3 clusters that provide the most commercial value and WHY.
Examples:
- best for sales targeting
- best for enterprise benchmarking
- best for compliance screening

### 5) Recommended Usage Strategy
Give a short plan:
- How a buyer should use these clusters in a product
- What kind of dashboard / workflow can be built
- What additional features would improve reliability

---

## Part C — General Summary of the Clustering (Final Section)
Write a strong concluding summary with:

### Overall Takeaways (5 bullets)
- What the clustering reveals about the dataset
- Any strong structure in the population
- How “noise” vs “stable clusters” might exist

### Limitations & Next Steps (5 bullets)
Must mention:
- scaling effects
- one-hot limitations
- missingness
- cluster size sensitivity
- need for validation (e.g. sampling companies per cluster)

==========================
CLUSTER CARDS (INPUT JSON)
==========================
{json.dumps(cards, indent=2)}
""".strip()



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

    profiles = pd.read_csv(PROFILES_PATH, index_col=0).T
    profiles.index = profiles.index.astype(int)

    print("Loaded profiles:", profiles.shape)

    cluster_cards = make_cluster_cards(profiles, top_k=8)
    cards_for_llm = compress_cards_for_llm(
        cluster_cards[:N_CLUSTERS_TO_SUMMARIZE],
        top_k=TOP_K_PER_SIDE_FOR_LLM
    )

    prompt = build_prompt(
        cards_for_llm,
        extra_context="The goal is to prove commercial value of dataset and show within + cross cluster differences.",
    )

    client = genai.Client(api_key=api_key)
    final_report = call_gemini_with_retries(client, prompt)

    out_path = Path(__file__).resolve().parent / "cluster_analysis.txt"
    out_path.write_text(final_report, encoding="utf-8")

    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
