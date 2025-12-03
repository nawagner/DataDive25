# Combined AI Users Chloropleth Design

## Overview

Interactive chloropleth map showing estimated AI users (Claude + ChatGPT) per country, with toggles for time period, metric type, and platform breakdown.

## Data Sources

| Source | Fields Used | Location |
|--------|-------------|----------|
| Anthropic Economic Index | `usage_count`, `gdp_per_working_age_capita`, `geo_id`, `geo_name` | HuggingFace: `Anthropic/EconomicIndex` |
| ChatGPT WAU by GDP | `gdp_per_capita_thousands_usd`, `median_wau_share_internet_users`, `time_period` | Local: `data/wau_share_by_gdp.csv` |
| Global Findex Database | `codewb`, `pop_adult`, `internet` | HuggingFace: `stablefusiondance/WorldBankDataDive2025` |

## Estimation Methodology

### Claude Users
Direct from Anthropic data:
- Filter: `facet == 'country'` and `variable == 'usage_count'`
- Exclude: `geo_name == 'not_classified'`

### ChatGPT Users (Estimated)
1. Get country's GDP per capita from Anthropic data
2. Interpolate WAU share from GDP→WAU curve (linear interpolation between buckets)
3. Get internet users: `pop_adult × internet` from Findex
4. Calculate: `chatgpt_users = internet_users × wau_share`

### Country Matching
- Anthropic uses ISO-2 (`geo_id`)
- Findex uses ISO-3 (`codewb`)
- Convert ISO-2 → ISO-3 using `pycountry` or embedded mapping

## Interactive Controls

| Control | Type | Options |
|---------|------|---------|
| Time Period | Toggle | May 2024 / May 2025 |
| Metric | Dropdown | Absolute users / Per capita / Per internet user |
| Platform | Radio | Combined / Claude only / ChatGPT only |

## Output Files

- `Team_Projects/AI_Readiness_Measures/app.py` - Streamlit dashboard
- `Team_Projects/AI_Readiness_Measures/load_ai_users_data.py` - Data loading script

## Technology Stack

- Streamlit for dashboard
- Plotly for choropleth map
- Pandas for data processing
- Data fetched from HuggingFace at runtime (not stored in git)
