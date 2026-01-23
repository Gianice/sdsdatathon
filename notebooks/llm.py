import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
# Ensure this matches your actual file name
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
    "Suzhou": "Major High-Tech Manufacturing Hub",
    "Beijing": "Political Capital & Corporate HQ Hub",
    "Shanghai": "Global Financial & Commercial Hub",
    "Shenzhen": "Tech & Innovation Hub",
    "Guangzhou": "Trade & Logistics Hub",
    "Hangzhou": "E-Commerce & Digital Tech Hub",
    "Chengdu": "Western China Economic Center",
    "Urumqi": "Western Regional Logistics Hub",
    "Kunming": "Gateway to Southeast Asia",
    "Xi'an": "Central Research & Industrial Hub",
    "Zhengzhou": "Major Transportation & Logistics Node"
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

    def _get_deviation_text(self, value, baseline, label):
        if pd.isna(value) or baseline == 0: return "Unknown"
        ratio = value / baseline
        if ratio > 3.0: return f"Very High (>3x Avg)"
        if ratio > 1.5: return f"High ({ratio:.1f}x Avg)"
        if ratio < 0.3: return f"Very Low (<0.3x Avg)"
        if ratio < 0.8: return f"Below Average"
        return "Average"

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
        
        # Operational Status Classifier
        op_status = "Unknown"
        op_note = ""
        
        if rev > 1000:
            op_status = "Active Commercial Entity"
            op_note = "Standard operational profile with reported revenue."
        elif emp > 0:
            op_status = "Active Branch / Non-Reporting"
            op_note = "Likely a cost center or branch office. Staff exists, but revenue is consolidated elsewhere."
        else:
            op_status = "Shell / Inactive Entity"
            op_note = "High risk. No reported revenue or staff. Likely a holding vehicle or dormant registration."

        web_col = self.col_map["website"]
        if web_col in sub.columns:
            website_pct = sub[web_col].notna().mean()
        else:
            website_pct = 0.0

        distinctive_traits = self._get_top_distinctive_features(sub)
        city = safe_mode("City")
        city_desc = CITY_CONTEXT.get(city, "Regional City")

        return {
            "id": int(cluster_id),
            "size": len(sub),
            "operational_status": op_status,
            "status_context": op_note,
            "identity": {
                "dominant_industry": safe_mode("SIC Description"),
                "dominant_location": f"{city}, {safe_mode('Country')} ({city_desc})",
                "structure": safe_mode("Entity Type")
            },
            "financial_profile": {
                "revenue": f"${rev:,.0f}",
                "employees": f"{int(emp)}",
            },
            "tech_profile": {
                "it_spend": f"${it_spend:,.0f}",
                "digital_maturity": f"{int(website_pct*100)}% have websites"
            },
            "analyst_notes": {
                "key_differentiators": distinctive_traits
            }
        }

# ---------------------------------------------------------
# 3. GENERATIVE ENGINE (LLM)
# ---------------------------------------------------------

def safe_generate(client, prompt):
    """Wraps API call with error handling to prevent crashes."""
    try:
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt,
            config={
                
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
                ]
            }
        )
        if response.text:
            return response.text
        else:
            return "*(Analysis unavailable for this batch due to content filters)*"
    except Exception as e:
        print(f"  [!] API Warning: {e}")
        return f"*(Error processing batch: {e})*"

def generate_landscape(client, all_contexts):
    prompt = f"""
    ### ROLE
    You are a Chief Data Strategist.
    
    ### DATA
    Summary of market segments (Sorted by Revenue):
    {json.dumps([c for c in all_contexts], indent=2)} 

    ### TASK
    Write an **Executive Dataset Overview** (Markdown).
    
    1. **Headline**: Summarize the mix of Active Commercial vs. Branch/Shell entities.
    2. **Strategic Segmentation**: Table with columns:
       | Cluster ID | Segment Name | Status (Active/Branch/Shell) | Primary Industry | Best Sales Use Case |
    3. **Spotlight**: Identify the best cluster for High-Ticket Sales.
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
    *(e.g., "Cluster 15: Urumqi Telecom Branches (Active Branch / Non-Reporting)")*
    
    **1. The Profile**
    * **Who:** [Identity -> dominant_industry] in [Identity -> dominant_location].
    * **Nature:** [Status Context] (Explain why revenue/staff might be zero).
    
    **2. The Signals**
    * **Distinctive Traits:** [Analyst Notes -> key_differentiators].
    * **Tech Maturity:** [Tech Profile -> digital_maturity] & Spend: [Tech Profile -> it_spend].
    
    **3. Commercial Verdict**
    * **Opportunity:** - If Active: What software to sell?
      - If Branch: Sell "Operational Efficiency" or "HQ Connection".
      - If Shell: "Data Cleansing" or "Risk Avoidance".
    * **Risk Level:** Low/Med/High.
    
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
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env")
        return
        
    client = genai.Client(api_key=api_key)

    print(f"Loading data from {RAW_DATA_PATH}...")
    try:
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: File not found. Please ensure {RAW_DATA_PATH} exists.")
        return

    if 'cluster_id' not in df.columns:
        print("ERROR: 'cluster_id' column missing. Did you save the clustering results?")
        return
    
    # Filter noise
    df = df[df['cluster_id'] != -1]

    print(f"Analyzing {len(df['cluster_id'].unique())} clusters...")
    analyzer = ClusterAnalyzer(df)
    unique_ids = sorted(df['cluster_id'].unique())
    
    all_contexts = []
    for cid in unique_ids:
        ctx = analyzer.build_context(cid)
        if ctx: all_contexts.append(ctx)
    
    # Sort: Commercial Entities First -> Then Branches -> Then Shells
    print(f"Found {len(all_contexts)} valid clusters.")
    try:
        all_contexts.sort(
            key=lambda x: (
                0 if x['operational_status'] == "Active Commercial Entity" else 1, 
                float(x['financial_profile']['revenue'].replace('$','').replace(',','')) * -1
            )
        )
    except:
        pass

    landscape_text = generate_landscape(client, all_contexts)
    
    cards_text = "## Detailed Segment Intelligence\n\n"
    for i in range(0, len(all_contexts), BATCH_SIZE):
        batch = all_contexts[i : i + BATCH_SIZE]
        
       
        batch_result = generate_deep_dives(client, batch)
        if batch_result:
            cards_text += batch_result + "\n\n"
        else:
            cards_text += "\n\n*(Batch skipped due to API error)*\n\n"
        
        time.sleep(2)

    full_report = f"# Champions Group Data Intelligence\n\n{landscape_text}\n\n{cards_text}"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"SUCCESS: Report saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()