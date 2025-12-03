# AI Users Chloropleth Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an interactive chloropleth showing combined Claude + ChatGPT users per country with toggles for time period, metric, and platform.

**Architecture:** Streamlit dashboard fetches data from HuggingFace at runtime, joins Anthropic (Claude users, GDP) with Findex (internet users), estimates ChatGPT users via WAU-by-GDP interpolation, renders Plotly choropleth with interactive controls.

**Tech Stack:** Streamlit, Plotly, Pandas, pycountry (ISO code conversion)

---

### Task 1: Create Data Loading Module

**Files:**
- Create: `Team_Projects/AI_Readiness_Measures/load_ai_users_data.py`

**Step 1: Write the ISO-2 to ISO-3 conversion function**

```python
#!/usr/bin/env python3
"""
Data loading for AI Users Chloropleth.

Fetches and joins:
- Anthropic Economic Index (Claude users, GDP per capita)
- Global Findex Database (internet users per country)
- ChatGPT WAU share by GDP (for estimation)
"""

import pandas as pd
import requests
from io import StringIO
from pathlib import Path

# HuggingFace URLs
ANTHROPIC_URL = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/release_2025_09_15/data/output/aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv"
FINDEX_URL = "https://huggingface.co/datasets/stablefusiondance/WorldBankDataDive2025/resolve/main/GlobalFindexDatabase2025.csv"

DATA_DIR = Path(__file__).parent / "data"


def iso2_to_iso3(iso2: str) -> str | None:
    """Convert ISO-2 country code to ISO-3."""
    # Common mappings (covers most countries in the datasets)
    ISO2_TO_ISO3 = {
        "AF": "AFG", "AL": "ALB", "DZ": "DZA", "AR": "ARG", "AM": "ARM",
        "AU": "AUS", "AT": "AUT", "AZ": "AZE", "BH": "BHR", "BD": "BGD",
        "BY": "BLR", "BE": "BEL", "BJ": "BEN", "BO": "BOL", "BA": "BIH",
        "BW": "BWA", "BR": "BRA", "BG": "BGR", "BF": "BFA", "BI": "BDI",
        "KH": "KHM", "CM": "CMR", "CA": "CAN", "CF": "CAF", "TD": "TCD",
        "CL": "CHL", "CN": "CHN", "CO": "COL", "CG": "COG", "CD": "COD",
        "CR": "CRI", "CI": "CIV", "HR": "HRV", "CY": "CYP", "CZ": "CZE",
        "DK": "DNK", "DO": "DOM", "EC": "ECU", "EG": "EGY", "SV": "SLV",
        "EE": "EST", "ET": "ETH", "FI": "FIN", "FR": "FRA", "GA": "GAB",
        "GE": "GEO", "DE": "DEU", "GH": "GHA", "GR": "GRC", "GT": "GTM",
        "GN": "GIN", "HT": "HTI", "HN": "HND", "HK": "HKG", "HU": "HUN",
        "IS": "ISL", "IN": "IND", "ID": "IDN", "IR": "IRN", "IQ": "IRQ",
        "IE": "IRL", "IL": "ISR", "IT": "ITA", "JM": "JAM", "JP": "JPN",
        "JO": "JOR", "KZ": "KAZ", "KE": "KEN", "KR": "KOR", "KW": "KWT",
        "KG": "KGZ", "LA": "LAO", "LV": "LVA", "LB": "LBN", "LR": "LBR",
        "LY": "LBY", "LT": "LTU", "LU": "LUX", "MG": "MDG", "MW": "MWI",
        "MY": "MYS", "ML": "MLI", "MT": "MLT", "MR": "MRT", "MU": "MUS",
        "MX": "MEX", "MD": "MDA", "MN": "MNG", "ME": "MNE", "MA": "MAR",
        "MZ": "MOZ", "MM": "MMR", "NA": "NAM", "NP": "NPL", "NL": "NLD",
        "NZ": "NZL", "NI": "NIC", "NE": "NER", "NG": "NGA", "NO": "NOR",
        "OM": "OMN", "PK": "PAK", "PA": "PAN", "PY": "PRY", "PE": "PER",
        "PH": "PHL", "PL": "POL", "PT": "PRT", "QA": "QAT", "RO": "ROU",
        "RU": "RUS", "RW": "RWA", "SA": "SAU", "SN": "SEN", "RS": "SRB",
        "SL": "SLE", "SG": "SGP", "SK": "SVK", "SI": "SVN", "ZA": "ZAF",
        "ES": "ESP", "LK": "LKA", "SD": "SDN", "SE": "SWE", "CH": "CHE",
        "SY": "SYR", "TW": "TWN", "TJ": "TJK", "TZ": "TZA", "TH": "THA",
        "TG": "TGO", "TN": "TUN", "TR": "TUR", "UG": "UGA", "UA": "UKR",
        "AE": "ARE", "GB": "GBR", "US": "USA", "UY": "URY", "UZ": "UZB",
        "VE": "VEN", "VN": "VNM", "YE": "YEM", "ZM": "ZMB", "ZW": "ZWE",
    }
    return ISO2_TO_ISO3.get(iso2.upper())
```

**Step 2: Add data fetching functions**

Add to the same file after the ISO mapping:

```python
def load_anthropic_data() -> pd.DataFrame:
    """Load Anthropic Economic Index data from HuggingFace."""
    print("Downloading Anthropic Economic Index data...")
    response = requests.get(ANTHROPIC_URL, timeout=120)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    print(f"  Downloaded {len(df):,} rows")
    return df


def load_findex_data() -> pd.DataFrame:
    """Load Global Findex Database from HuggingFace."""
    print("Downloading Global Findex Database...")
    response = requests.get(FINDEX_URL, timeout=120)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text), low_memory=False)
    print(f"  Downloaded {len(df):,} rows")
    return df


def load_wau_data() -> pd.DataFrame:
    """Load ChatGPT WAU share by GDP data."""
    wau_path = DATA_DIR / "wau_share_by_gdp.csv"
    return pd.read_csv(wau_path)
```

**Step 3: Add Claude users extraction**

```python
def get_claude_users(anthropic_df: pd.DataFrame) -> pd.DataFrame:
    """Extract Claude usage counts by country."""
    # Filter for country-level usage counts
    country_usage = anthropic_df[
        (anthropic_df['facet'] == 'country') &
        (anthropic_df['variable'] == 'usage_count') &
        (anthropic_df['geo_name'] != 'not_classified')
    ].copy()

    # Convert ISO-2 to ISO-3
    country_usage['iso3'] = country_usage['geo_id'].apply(iso2_to_iso3)
    country_usage = country_usage[country_usage['iso3'].notna()]

    return country_usage[['iso3', 'geo_name', 'value']].rename(columns={
        'geo_name': 'country_name',
        'value': 'claude_users'
    })


def get_gdp_per_capita(anthropic_df: pd.DataFrame) -> pd.DataFrame:
    """Extract GDP per working-age capita by country."""
    gdp_data = anthropic_df[
        (anthropic_df['facet'] == 'country') &
        (anthropic_df['variable'] == 'gdp_per_working_age_capita') &
        (anthropic_df['geo_name'] != 'not_classified')
    ].copy()

    gdp_data['iso3'] = gdp_data['geo_id'].apply(iso2_to_iso3)
    gdp_data = gdp_data[gdp_data['iso3'].notna()]

    return gdp_data[['iso3', 'value']].rename(columns={
        'value': 'gdp_per_capita'
    })
```

**Step 4: Add ChatGPT estimation function**

```python
def interpolate_wau_share(gdp_k: float, wau_df: pd.DataFrame, time_period: str = "May 2025") -> float:
    """Interpolate WAU share from GDP per capita using the WAU-by-GDP curve."""
    period_data = wau_df[wau_df['time_period'] == time_period].sort_values('gdp_per_capita_thousands_usd')

    if period_data.empty:
        return 0.0

    gdp_values = period_data['gdp_per_capita_thousands_usd'].values
    wau_values = period_data['median_wau_share_internet_users'].values

    # Clamp to range
    if gdp_k <= gdp_values[0]:
        return wau_values[0]
    if gdp_k >= gdp_values[-1]:
        return wau_values[-1]

    # Linear interpolation
    for i in range(len(gdp_values) - 1):
        if gdp_values[i] <= gdp_k <= gdp_values[i + 1]:
            t = (gdp_k - gdp_values[i]) / (gdp_values[i + 1] - gdp_values[i])
            return wau_values[i] + t * (wau_values[i + 1] - wau_values[i])

    return wau_values[-1]


def estimate_chatgpt_users(
    gdp_df: pd.DataFrame,
    findex_df: pd.DataFrame,
    wau_df: pd.DataFrame,
    time_period: str = "May 2025"
) -> pd.DataFrame:
    """Estimate ChatGPT users per country."""
    # Get 2024 Findex data for 'all' group
    findex_2024 = findex_df[
        (findex_df['year'] == 2024) &
        (findex_df['group'] == 'all') &
        (findex_df['internet'].notna())
    ][['codewb', 'countrynewwb', 'pop_adult', 'internet']].copy()

    findex_2024.rename(columns={'codewb': 'iso3', 'countrynewwb': 'country_name_findex'}, inplace=True)

    # Calculate internet users
    findex_2024['internet_users'] = findex_2024['pop_adult'] * findex_2024['internet']

    # Join with GDP data
    merged = findex_2024.merge(gdp_df, on='iso3', how='inner')

    # Convert GDP to thousands for WAU lookup
    merged['gdp_k'] = merged['gdp_per_capita'] / 1000

    # Estimate ChatGPT users
    merged['wau_share'] = merged['gdp_k'].apply(
        lambda x: interpolate_wau_share(x, wau_df, time_period)
    )
    merged['chatgpt_users'] = merged['internet_users'] * merged['wau_share']

    return merged[['iso3', 'country_name_findex', 'pop_adult', 'internet_users', 'chatgpt_users']]
```

**Step 5: Add main data loading function**

```python
def load_ai_users_data(time_period: str = "May 2025") -> pd.DataFrame:
    """
    Load and combine all data sources into a single DataFrame.

    Returns DataFrame with columns:
    - iso3: ISO-3 country code
    - country_name: Country name
    - claude_users: Actual Claude usage count
    - chatgpt_users: Estimated ChatGPT users
    - total_ai_users: Sum of Claude + ChatGPT
    - pop_adult: Adult population
    - internet_users: Number of internet users
    - ai_users_per_capita: Total AI users / adult population
    - ai_users_per_internet: Total AI users / internet users
    """
    # Load all data sources
    anthropic_df = load_anthropic_data()
    findex_df = load_findex_data()
    wau_df = load_wau_data()

    # Extract Claude users and GDP
    claude_df = get_claude_users(anthropic_df)
    gdp_df = get_gdp_per_capita(anthropic_df)

    # Estimate ChatGPT users
    chatgpt_df = estimate_chatgpt_users(gdp_df, findex_df, wau_df, time_period)

    # Merge Claude and ChatGPT data
    combined = claude_df.merge(chatgpt_df, on='iso3', how='outer')

    # Fill missing values
    combined['claude_users'] = combined['claude_users'].fillna(0)
    combined['chatgpt_users'] = combined['chatgpt_users'].fillna(0)

    # Use Claude country name where available, else Findex
    combined['country_name'] = combined['country_name'].fillna(combined['country_name_findex'])

    # Calculate totals and per-capita metrics
    combined['total_ai_users'] = combined['claude_users'] + combined['chatgpt_users']
    combined['ai_users_per_capita'] = combined['total_ai_users'] / combined['pop_adult']
    combined['ai_users_per_internet'] = combined['total_ai_users'] / combined['internet_users']

    # Clean up
    result = combined[[
        'iso3', 'country_name', 'claude_users', 'chatgpt_users',
        'total_ai_users', 'pop_adult', 'internet_users',
        'ai_users_per_capita', 'ai_users_per_internet'
    ]].copy()

    # Remove rows with no AI users
    result = result[result['total_ai_users'] > 0]

    print(f"\nCombined data: {len(result)} countries")
    return result


if __name__ == "__main__":
    # Test data loading
    df = load_ai_users_data("May 2025")
    print("\nTop 10 countries by total AI users:")
    print(df.nlargest(10, 'total_ai_users')[['country_name', 'claude_users', 'chatgpt_users', 'total_ai_users']])
```

**Step 6: Run and verify data loading works**

Run: `cd Team_Projects/AI_Readiness_Measures && python load_ai_users_data.py`

Expected: Downloads data, prints top 10 countries by AI users

**Step 7: Commit**

```bash
git add Team_Projects/AI_Readiness_Measures/load_ai_users_data.py
git commit -m "feat: add AI users data loading module

Fetches Anthropic (Claude users, GDP) and Findex (internet users),
estimates ChatGPT users via WAU-by-GDP interpolation."
```

---

### Task 2: Create Streamlit Dashboard

**Files:**
- Create: `Team_Projects/AI_Readiness_Measures/app.py`

**Step 1: Write dashboard boilerplate with page config**

```python
#!/usr/bin/env python3
"""
AI Users Chloropleth Dashboard

Interactive map showing estimated AI users (Claude + ChatGPT) per country.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from load_ai_users_data import load_ai_users_data

# Page configuration
st.set_page_config(
    page_title="AI Users by Country",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

**Step 2: Add data loading with caching**

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_data(time_period: str) -> pd.DataFrame:
    """Load AI users data with caching."""
    return load_ai_users_data(time_period)
```

**Step 3: Add choropleth creation function**

```python
def create_choropleth(
    df: pd.DataFrame,
    metric: str,
    platform: str,
    title: str
) -> go.Figure:
    """Create choropleth map with specified metric and platform filter."""

    # Select value column based on metric and platform
    if platform == "Combined":
        if metric == "Absolute users":
            value_col = "total_ai_users"
            color_label = "Total AI Users"
        elif metric == "Per capita":
            value_col = "ai_users_per_capita"
            color_label = "AI Users per Capita"
        else:  # Per internet user
            value_col = "ai_users_per_internet"
            color_label = "AI Users per Internet User"
    elif platform == "Claude only":
        value_col = "claude_users"
        color_label = "Claude Users"
        if metric == "Per capita":
            df = df.copy()
            df['claude_per_capita'] = df['claude_users'] / df['pop_adult']
            value_col = "claude_per_capita"
            color_label = "Claude Users per Capita"
        elif metric == "Per internet user":
            df = df.copy()
            df['claude_per_internet'] = df['claude_users'] / df['internet_users']
            value_col = "claude_per_internet"
            color_label = "Claude Users per Internet User"
    else:  # ChatGPT only
        value_col = "chatgpt_users"
        color_label = "ChatGPT Users (Est.)"
        if metric == "Per capita":
            df = df.copy()
            df['chatgpt_per_capita'] = df['chatgpt_users'] / df['pop_adult']
            value_col = "chatgpt_per_capita"
            color_label = "ChatGPT Users per Capita (Est.)"
        elif metric == "Per internet user":
            df = df.copy()
            df['chatgpt_per_internet'] = df['chatgpt_users'] / df['internet_users']
            value_col = "chatgpt_per_internet"
            color_label = "ChatGPT Users per Internet User (Est.)"

    # Create hover text
    df = df.copy()
    df['hover_text'] = df.apply(
        lambda r: f"<b>{r['country_name']}</b><br>" +
                  f"Claude: {r['claude_users']:,.0f}<br>" +
                  f"ChatGPT (est.): {r['chatgpt_users']:,.0f}<br>" +
                  f"Total: {r['total_ai_users']:,.0f}",
        axis=1
    )

    # Color scale (blue-green like Anthropic style)
    colorscale = [
        [0, '#e8f4f8'],
        [0.2, '#c6e5e8'],
        [0.4, '#9dd3d8'],
        [0.6, '#6bb6c1'],
        [0.8, '#3a9ab0'],
        [1, '#1a7a8f']
    ]

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locations=df['iso3'],
        z=df[value_col],
        text=df['hover_text'],
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(text=color_label, font=dict(size=12)),
            thickness=20,
            len=0.6,
        ),
        hovertemplate='%{text}<extra></extra>',
        marker=dict(line=dict(width=0.5, color='rgba(255,255,255,0.8)'))
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#1a1a1a'),
            x=0.5,
            xanchor='center'
        ),
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='rgba(200,200,200,0.3)',
            showland=True,
            landcolor='rgba(250,250,250,1)',
            showocean=True,
            oceancolor='rgba(240,248,255,1)',
            projection_type='natural earth',
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    return fig
```

**Step 4: Add main app function**

```python
def main():
    """Main Streamlit app."""
    st.title("🌍 AI Users by Country")
    st.markdown("""
    Interactive chloropleth showing estimated AI users (Claude + ChatGPT) per country.

    - **Claude users**: Actual usage data from Anthropic Economic Index
    - **ChatGPT users**: Estimated from WAU share by GDP × internet users
    """)

    # Sidebar controls
    st.sidebar.header("Controls")

    # Time period toggle
    time_period = st.sidebar.radio(
        "Time Period (ChatGPT WAU data)",
        ["May 2025", "May 2024"],
        help="Affects ChatGPT user estimates (WAU share changes over time)"
    )

    # Metric selector
    metric = st.sidebar.selectbox(
        "Metric",
        ["Absolute users", "Per capita", "Per internet user"],
        help="How to measure AI adoption"
    )

    # Platform selector
    platform = st.sidebar.radio(
        "Platform",
        ["Combined", "Claude only", "ChatGPT only"],
        help="Which AI platform(s) to show"
    )

    # Load data
    with st.spinner("Loading data from HuggingFace..."):
        df = get_data(time_period)

    # Create title based on selections
    title_parts = []
    if platform == "Combined":
        title_parts.append("AI Users (Claude + ChatGPT)")
    elif platform == "Claude only":
        title_parts.append("Claude Users")
    else:
        title_parts.append("ChatGPT Users (Estimated)")

    if metric == "Per capita":
        title_parts.append("per Capita")
    elif metric == "Per internet user":
        title_parts.append("per Internet User")

    title_parts.append(f"({time_period})")
    title = " ".join(title_parts)

    # Create and display map
    fig = create_choropleth(df, metric, platform, title)
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Countries", len(df))
    with col2:
        st.metric("Total Claude Users", f"{df['claude_users'].sum():,.0f}")
    with col3:
        st.metric("Est. ChatGPT Users", f"{df['chatgpt_users'].sum():,.0f}")
    with col4:
        st.metric("Total AI Users", f"{df['total_ai_users'].sum():,.0f}")

    # Top countries table
    with st.expander("📋 Top 20 Countries by Total AI Users"):
        top_20 = df.nlargest(20, 'total_ai_users')[[
            'country_name', 'claude_users', 'chatgpt_users', 'total_ai_users',
            'ai_users_per_capita', 'ai_users_per_internet'
        ]].copy()
        top_20.columns = ['Country', 'Claude', 'ChatGPT (Est.)', 'Total', 'Per Capita', 'Per Internet User']
        st.dataframe(top_20, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Sources:**
    - Claude users: [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex)
    - Internet users: [Global Findex Database 2025](https://huggingface.co/datasets/stablefusiondance/WorldBankDataDive2025)
    - ChatGPT WAU share: World Bank DataDive 2025

    *ChatGPT users are estimates based on WAU share by GDP per capita applied to internet user counts.*
    """)


if __name__ == "__main__":
    main()
```

**Step 5: Run and verify dashboard works**

Run: `cd Team_Projects/AI_Readiness_Measures && streamlit run app.py`

Expected: Dashboard opens in browser with interactive map

**Step 6: Commit**

```bash
git add Team_Projects/AI_Readiness_Measures/app.py
git commit -m "feat: add AI users chloropleth Streamlit dashboard

Interactive map with toggles for time period, metric (absolute/per capita/per internet user), and platform (combined/Claude/ChatGPT)."
```

---

### Task 3: Add Requirements File

**Files:**
- Create: `Team_Projects/AI_Readiness_Measures/requirements.txt`

**Step 1: Create requirements file**

```
streamlit>=1.28.0
plotly>=5.18.0
pandas>=2.0.0
requests>=2.31.0
```

**Step 2: Commit**

```bash
git add Team_Projects/AI_Readiness_Measures/requirements.txt
git commit -m "chore: add requirements for AI users dashboard"
```

---

### Task 4: Update Project README

**Files:**
- Modify: `Team_Projects/AI_Readiness_Measures/project.md`

**Step 1: Read current project.md**

Read the file to understand existing structure.

**Step 2: Add dashboard documentation**

Add section about the new dashboard to the existing project.md.

**Step 3: Commit**

```bash
git add Team_Projects/AI_Readiness_Measures/project.md
git commit -m "docs: add AI users chloropleth to project documentation"
```

---

### Task 5: Final Verification and Push

**Step 1: Run the full pipeline**

```bash
cd Team_Projects/AI_Readiness_Measures
python load_ai_users_data.py  # Verify data loading
streamlit run app.py          # Verify dashboard
```

**Step 2: Git status and push**

```bash
git status
git push origin move-assets-to-ai-readiness
```

Expected: All commits pushed, dashboard functional
