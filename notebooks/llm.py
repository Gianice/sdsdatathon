import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from collections import defaultdict

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
# Ensure this filename matches your actual CSV
RAW_DATA_PATH = BASE_DIR / "../raw_data/champions_group_data_with_cluster.csv"
OUTPUT_PATH = BASE_DIR / "champions_cluster_intelligence.md"

#change model if exceed rate limit
#MODEL_ID = "gemini-3-flash-preview"
MODEL_ID = "gemini-2.5-flash"
#MODEL_ID = "gemini-2.5-flash-lite"

BATCH_SIZE = 5

# ---------------------------------------------------------
# 1. KNOWLEDGE BASES
# ---------------------------------------------------------
FEATURE_MAP = {
    "log_revenue_usd": "Revenue Scale",
    "log_employees_total": "Workforce Size",
    "log_it_spend": "IT Investment Level",
    "log_org_complexity_count": "Corporate Hierarchy Complexity",
    "credibility_score_norm": "Data Transparency Score",
    "has_website": "Digital Presence (Website)",
    "parent_foreign_flag": "International Ownership",
    "has_global_ultimate": "Part of Global Group",
    "it_spend_rate": "IT Budget Utilization",
    "pc_midpoint": "Hardware Asset Volume",
    "sic_code_count": "Diversification (Industry Codes)",
    "company_age": "Years in Business",
    "is_headquarters": "Headquarters Status",
    "is_domestic_ultimate": "Domestic Ultimate Status"
}

CITY_CONTEXT = {
    "Suzhou": "High-Tech Mfg Hub",
    "Beijing": "Capital & HQ Hub",
    "Shanghai": "Financial Hub",
    "Shenzhen": "Innovation Hub",
    "Guangzhou": "Trade Hub",
    "Hangzhou": "E-Commerce Hub",
    "Chengdu": "Western Economic Center",
    "Urumqi": "Western Logistics Hub",
    "Kunming": "SE Asia Gateway",
    "Xi'an": "Research Hub",
    "Zhengzhou": "Transport Node"
}

# ---------------------------------------------------------
# 2. ANALYTICAL ENGINE
# ---------------------------------------------------------

class ClusterAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
        self.col_map = {
            "revenue": "Revenue (USD)" if "Revenue (USD)" in df.columns else "revenue_usd",
            "employees": "Employees Total" if "Employees Total" in df.columns else "employees_total",
            "it_spend": "IT spend" if "IT spend" in df.columns else "it_spend", 
            "website": "Website" if "Website" in df.columns else "has_website"
        }
        
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.numeric_cols = [c for c in self.numeric_cols if "cluster" not in c and "id" not in c.lower()]
        
        self.global_stats = {
            "median_revenue": self.df[self.col_map["revenue"]].median(),
            "median_employees": self.df[self.col_map["employees"]].median(),
            "median_it_spend": self.df[self.col_map["it_spend"]].median(),
        }
        self.global_means = self.df[self.numeric_cols].mean()

    def _get_multiplier(self, value, baseline):
        if baseline == 0: return 0
        return round(value / baseline, 1)

    def _get_top_distinctive_features(self, cluster_df, top_k=4):
        if cluster_df.empty: return []
        cluster_means = cluster_df[self.numeric_cols].mean()
        c_means, g_means = cluster_means.align(self.global_means, join='inner')
        diff = (c_means - g_means) / (g_means.replace(0, 1))
        
        top_features = diff.sort_values(ascending=False).head(top_k)
        
        readable_features = []
        for feat, score in top_features.items():
            if "cluster" in feat.lower() or "id" in feat.lower(): continue
            if score > 0.5: 
                human_name = FEATURE_MAP.get(feat, feat)
                readable_features.append(f"{human_name} is distinctively high")
        return readable_features

    def build_context(self, cluster_id: int) -> dict:
        sub = self.df[self.df["cluster_id"] == cluster_id]
        if len(sub) < 5: return None 

        def safe_mode(col): 
            if col not in sub.columns: return "Unknown"
            return sub[col].mode()[0] if not sub[col].mode().empty else "Unknown"
            
        rev = sub[self.col_map["revenue"]].median()
        emp = sub[self.col_map["employees"]].median()
        it_spend = sub[self.col_map["it_spend"]].median()
        
        op_status = "Unknown"
        status_rank = 3
        
        if rev > 1000:
            op_status = "Active Commercial Entity"
            status_rank = 0
        elif emp > 0:
            op_status = "Active Branch / Cost Center"
            status_rank = 1
        else:
            op_status = "Shell / Inactive Entity"
            status_rank = 2

        web_col = self.col_map["website"]
        website_pct = sub[web_col].notna().mean() if web_col in sub.columns else 0.0

        distinctive_traits = self._get_top_distinctive_features(sub)
        city = safe_mode("City")
        city_desc = CITY_CONTEXT.get(city, "Regional City")
        industry = safe_mode("SIC Description")

        rev_mult = self._get_multiplier(rev, self.global_stats['median_revenue'])
        emp_mult = self._get_multiplier(emp, self.global_stats['median_employees'])

        return {
            "id": int(cluster_id),
            "size": len(sub),
            "_sort_status": status_rank, 
            "_sort_revenue": rev,
            "_group_industry": industry,
            
            "operational_status": op_status,
            "identity": {
                "dominant_industry": industry,
                "dominant_location": f"{city}, {safe_mode('Country')} ({city_desc})",
                "structure": safe_mode("Entity Type")
            },
            "comparative_metrics": {
                "revenue_vs_global": f"{rev_mult}x Global Avg" if rev_mult > 0 else "Zero Revenue",
                "employees_vs_global": f"{emp_mult}x Global Avg" if emp_mult > 0 else "Zero Staff",
                "raw_revenue": f"${rev:,.0f}",
                "raw_it_spend": f"${it_spend:,.0f}"
            },
            "tech_profile": {
                "digital_maturity": f"{int(website_pct*100)}% have websites",
                "it_spend": f"${it_spend:,.0f}"
            },
            "analyst_notes": {
                "key_differentiators": distinctive_traits
            }
        }

# ---------------------------------------------------------
# 3. GENERATIVE ENGINE (LLM)
# ---------------------------------------------------------

def safe_generate(client, prompt):
    try:
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt,
            config={"safety_settings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]}
        )
        return response.text if response.text else "*(Analysis skipped)*"
    except Exception as e:
        return f"*(Error: {e})*"

def generate_landscape(client, all_contexts, grouped_insights):
    """
    Generates the Executive Summary.
    Uses 'Active Commercial' clusters first for strategic importance.
    """
    prompt = f"""
    ### ROLE
    You are a Chief Data Strategist presenting to a Private Equity firm.
    
    ### CONTEXT: SIMILARITY ANALYSIS
    We have grouped the clusters by industry to help you find patterns:
    {grouped_insights}

    ### DATA INPUT
    Summary of top segments (Sorted by Importance/Revenue):
    {json.dumps([c for c in all_contexts if c['_sort_status'] <= 1][:20], indent=2)} 

    ### TASK
    Write an **Executive Dataset Overview** (Markdown).
    
    1. **Strategic Segmentation Matrix**: Create a table with these columns:
       | Operational Tier | Primary Industries | Cluster IDs | Commercial Strategy |
       *(Group similar clusters together in rows. e.g. "Construction" -> "28, 29, 41")*

    2. **Comparative Analysis**:
       * **The Whales vs. The Long Tail**: Compare the "Active Commercial" segments against the "Branch/Shell" segments.
       * **Geographic Variance**: Briefly mention the mix of Hubs (Beijing/Shanghai) vs Regional cities.

    3. **Spotlight**: Identify the single best cluster for **High-Ticket Sales**.
    """
    print("Generating Executive Landscape...")
    return safe_generate(client, prompt)

def generate_deep_dives(client, batch_contexts):
    prompt = f"""
    ### ROLE
    You are a Private Equity Analyst.
    
    ### INPUT
    {json.dumps(batch_contexts, indent=2)}
    
    ### TASK
    Write a **Commercial Intelligence Card** for each cluster.
    
    ### FORMAT (Markdown)
    #### Cluster [ID]: [Name] ([Operational Status])
    *(e.g., "Cluster 28: Zhengzhou Construction (Active Commercial)")*
    
    **1. The Profile**
    * **Who:** [Identity -> dominant_industry] in [Identity -> dominant_location].
    * **Vs. Benchmark:** [comparative_metrics -> revenue_vs_global] and [comparative_metrics -> employees_vs_global].
    
    **2. The Signals**
    * **Key Differentiator:** [Analyst Notes -> key_differentiators].
    * **Tech Maturity:** [Tech Profile -> digital_maturity] (Spend: [comparative_metrics -> raw_it_spend]).
    
    **3. Commercial Verdict**
    * **Strategy:** [Specific Product to Sell] or [Specific Risk to Avoid].
    * **Similar Clusters:** (Mention if this looks like other clusters based on Industry).
    
    ---
    """
    print(f"Generating Deep Dives for batch...")
    return safe_generate(client, prompt)

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    print(f"Loading data from {RAW_DATA_PATH}...")
    try:
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    if 'cluster_id' in df.columns: df = df[df['cluster_id'] != -1]

    print("Analyzing clusters...")
    analyzer = ClusterAnalyzer(df)
    unique_ids = sorted(df['cluster_id'].unique())
    
    all_contexts = []
    industry_groups = defaultdict(list)

    for cid in unique_ids:
        ctx = analyzer.build_context(cid)
        if ctx: 
            all_contexts.append(ctx)
            industry_groups[ctx['_group_industry']].append(cid)

    # Convert groups to string for Landscape Prompt
    grouped_insights_str = "Identified Industry Groups:\n"
    for ind, cids in industry_groups.items():
        if len(cids) > 1:
            grouped_insights_str += f"- {ind}: Clusters {cids}\n"

    # --- 1. SORT FOR LANDSCAPE (By Importance) ---
    contexts_by_importance = sorted(all_contexts, key=lambda x: (x['_sort_status'], x['_sort_revenue'] * -1))
    
    # --- 2. SORT FOR DEEP DIVES (By Cluster ID 1, 2, 3...) ---
    contexts_by_id = sorted(all_contexts, key=lambda x: x['id'])

    # Generate Landscape
    landscape_text = generate_landscape(client, contexts_by_importance, grouped_insights_str)
    
    # Generate Deep Dives 
    cards_text = "## Detailed Segment Intelligence\n\n"
    for i in range(0, len(contexts_by_id), BATCH_SIZE):
        batch = contexts_by_id[i : i + BATCH_SIZE]
        result = generate_deep_dives(client, batch)
        cards_text += result + "\n\n"
        time.sleep(2)

    full_report = f"# Champions Group Data Intelligence\n\n{landscape_text}\n\n{cards_text}"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"SUCCESS: Report saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()