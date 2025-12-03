#!/usr/bin/env python3
"""
Anthropic Economic Index - Country Trends Analysis

This script downloads data from Anthropic's Economic Index dataset and
visualizes trends by country for AI usage patterns.

Data source: https://huggingface.co/datasets/Anthropic/EconomicIndex
"""

import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
import os

# Configuration
DATA_URL = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/release_2025_09_15/data/output/aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv"


def download_data():
    """Download the Economic Index dataset from HuggingFace."""
    print("Downloading Economic Index data from HuggingFace...")
    print(f"URL: {DATA_URL}")

    response = requests.get(DATA_URL, timeout=120)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    print(f"Downloaded {len(df):,} rows")

    return df


def explore_data(df):
    """Print basic info about the dataset."""
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")

    print(f"\nDate range: {df['date_start'].min()} to {df['date_end'].max()}")

    print(f"\nFacets available: {df['facet'].unique().tolist()}")

    print(f"\nVariables available: {df['variable'].unique().tolist()}")

    # Countries - use geo_name for actual country names
    country_data = df[df['facet'] == 'country']
    if not country_data.empty:
        countries = country_data['geo_name'].dropna().unique()
        print(f"\nCountries in dataset ({len(countries)}): {sorted(countries)[:20]}...")


def analyze_country_usage(df):
    """Analyze and visualize usage patterns by country using raw counts."""
    print("\n" + "="*60)
    print("COUNTRY USAGE ANALYSIS")
    print("="*60)

    # Filter for country-level data
    country_df = df[df['facet'] == 'country'].copy()
    # Filter out 'not_classified' entries
    country_df = country_df[country_df['geo_name'] != 'not_classified']

    if country_df.empty:
        print("No country-level data found")
        return

    # Get raw usage counts
    usage_count = country_df[country_df['variable'] == 'usage_count'].copy()
    usage_count = usage_count[usage_count['value'] > 0]
    usage_count = usage_count.sort_values('value', ascending=False)

    total_usage = usage_count['value'].sum()
    print(f"\nTotal usage count: {total_usage:,.0f}")

    print("\n--- Top 20 Countries by Raw Usage Count ---")
    top_20 = usage_count.head(20)[['geo_name', 'value']].copy()
    for _, row in top_20.iterrows():
        pct = (row['value'] / total_usage) * 100
        print(f"  {row['geo_name']:25s}  {row['value']:>10,.0f}  ({pct:5.2f}%)")

    # Plot raw usage counts
    plt.figure(figsize=(12, 8))
    plt.barh(top_20['geo_name'], top_20['value'], color='steelblue')
    plt.xlabel('Usage Count')
    plt.ylabel('Country')
    plt.title(f'Top 20 Countries by Claude.ai Raw Usage Count\n(Total: {total_usage:,.0f})')
    plt.gca().invert_yaxis()
    # Format x-axis with commas
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig('country_usage_count.png', dpi=150)
    plt.close()
    print("\nSaved: country_usage_count.png")

    # Get per-capita usage (raw, not index)
    per_capita = country_df[country_df['variable'] == 'usage_per_capita'].copy()
    per_capita = per_capita[per_capita['value'] > 0]
    per_capita = per_capita.sort_values('value', ascending=False)

    print("\n--- Top 20 Countries by Per-Capita Usage ---")
    print("(Usage count per person in working-age population)")
    top_20_pc = per_capita.head(20)[['geo_name', 'value']].copy()
    for _, row in top_20_pc.iterrows():
        # Convert to per million for readability
        per_million = row['value'] * 1_000_000
        print(f"  {row['geo_name']:25s}  {per_million:>10,.1f} per million")

    # Plot per-capita usage
    plt.figure(figsize=(12, 8))
    # Convert to per million for chart
    top_20_pc['per_million'] = top_20_pc['value'] * 1_000_000
    plt.barh(top_20_pc['geo_name'], top_20_pc['per_million'], color='teal')
    plt.xlabel('Usage per Million People (Working-Age Population)')
    plt.ylabel('Country')
    plt.title('Top 20 Countries by Per-Capita AI Usage')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('country_per_capita_usage.png', dpi=150)
    plt.close()
    print("\nSaved: country_per_capita_usage.png")


def analyze_task_patterns(df):
    """Analyze what tasks are most common globally using raw counts."""
    print("\n" + "="*60)
    print("TOP TASKS GLOBALLY")
    print("="*60)

    # Look for onet_task facet data
    task_df = df[df['facet'] == 'onet_task'].copy()

    if task_df.empty:
        print("No task data found")
        return

    # Get task counts
    task_count = task_df[task_df['variable'] == 'onet_task_count'].copy()

    if not task_count.empty:
        # Filter out 'not_classified' and 'none' entries
        task_count = task_count[
            ~task_count['cluster_name'].str.contains('not_classified', case=False, na=False) &
            (task_count['cluster_name'].str.lower() != 'none')
        ]
        task_count = task_count.sort_values('value', ascending=False)

        total_tasks = task_count['value'].sum()
        print(f"\nFound {len(task_count)} classified task categories")
        print(f"Total task count: {total_tasks:,.0f}")

        print("\n--- Top 15 Tasks by Raw Count ---")
        top_tasks = task_count.head(15)[['cluster_name', 'value']].copy()
        for _, row in top_tasks.iterrows():
            # Truncate long task names
            task_name = row['cluster_name'][:55] + '...' if len(row['cluster_name']) > 55 else row['cluster_name']
            pct = (row['value'] / total_tasks) * 100
            print(f"  {row['value']:>8,.0f}  ({pct:4.1f}%)  {task_name}")

        # Plot top tasks
        plt.figure(figsize=(14, 8))
        top_10_tasks = task_count.head(10)
        labels = [name[:45] + '...' if len(name) > 45 else name for name in top_10_tasks['cluster_name']]
        plt.barh(labels, top_10_tasks['value'], color='coral')
        plt.xlabel('Usage Count')
        plt.ylabel('Task Category')
        plt.title(f'Top 10 O*NET Tasks by Claude.ai Usage Count\n(Total: {total_tasks:,.0f})')
        plt.gca().invert_yaxis()
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.tight_layout()
        plt.savefig('top_tasks.png', dpi=150)
        plt.close()
        print("\nSaved: top_tasks.png")


def analyze_collaboration_patterns(df):
    """Analyze automation vs augmentation patterns."""
    print("\n" + "="*60)
    print("AUTOMATION vs AUGMENTATION ANALYSIS")
    print("="*60)

    # Look for collaboration_automation_augmentation facet
    collab_df = df[df['facet'] == 'collaboration_automation_augmentation'].copy()

    if collab_df.empty:
        print("No automation/augmentation pattern data found")
        return

    print(f"\nCollaboration patterns found: {collab_df['cluster_name'].unique().tolist()}")

    # Get automation and augmentation percentages
    auto_pct = collab_df[
        (collab_df['cluster_name'] == 'automation') &
        (collab_df['variable'] == 'automation_pct')
    ]
    aug_pct = collab_df[
        (collab_df['cluster_name'] == 'augmentation') &
        (collab_df['variable'] == 'augmentation_pct')
    ]

    if not auto_pct.empty and not aug_pct.empty:
        # Values are already in percentage form (e.g., 59.27 = 59.27%)
        auto_val = auto_pct['value'].mean()  # Average across all entries
        aug_val = aug_pct['value'].mean()
        print(f"\nGlobal Split (averaged across all categories):")
        print(f"  Automation (AI doing tasks independently): {auto_val:.1f}%")
        print(f"  Augmentation (AI assisting humans):        {aug_val:.1f}%")

        # Create pie chart
        plt.figure(figsize=(8, 6))
        sizes = [auto_val, aug_val]
        labels = ['Automation\n(AI independent)', 'Augmentation\n(AI assisting)']
        colors = ['#ff6b6b', '#4ecdc4']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.02, 0.02))
        plt.title('Claude.ai Usage: Automation vs Augmentation')
        plt.tight_layout()
        plt.savefig('automation_vs_augmentation.png', dpi=150)
        plt.close()
        print("\nSaved: automation_vs_augmentation.png")


def create_summary_dashboard(df):
    """Create a summary visualization dashboard using raw counts."""
    print("\n" + "="*60)
    print("CREATING SUMMARY DASHBOARD")
    print("="*60)

    country_df = df[df['facet'] == 'country'].copy()
    # Filter out not_classified
    country_df = country_df[country_df['geo_name'] != 'not_classified']

    if country_df.empty:
        print("No country data for dashboard")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Top 10 by raw usage count
    ax1 = axes[0, 0]
    usage_count = country_df[country_df['variable'] == 'usage_count'].sort_values('value', ascending=False).head(10)
    if not usage_count.empty:
        ax1.barh(usage_count['geo_name'], usage_count['value'], color='steelblue')
        ax1.set_xlabel('Usage Count')
        ax1.set_title('Top 10 Countries by Raw Usage Count')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax1.invert_yaxis()

    # 2. Distribution of usage counts (log scale)
    ax2 = axes[0, 1]
    all_usage = country_df[country_df['variable'] == 'usage_count']
    all_usage = all_usage[all_usage['value'] > 0]
    if not all_usage.empty:
        ax2.hist(all_usage['value'], bins=50, color='teal', edgecolor='white', log=True)
        ax2.set_xlabel('Usage Count')
        ax2.set_ylabel('Number of Countries (log scale)')
        ax2.set_title('Distribution of Usage Counts')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # 3. Top 10 by per-capita usage
    ax3 = axes[1, 0]
    per_capita = country_df[country_df['variable'] == 'usage_per_capita']
    per_capita = per_capita[per_capita['value'] > 0]
    if not per_capita.empty:
        top_pc = per_capita.sort_values('value', ascending=False).head(10)
        # Convert to per million
        ax3.barh(top_pc['geo_name'], top_pc['value'] * 1_000_000, color='green')
        ax3.set_xlabel('Usage per Million People')
        ax3.set_title('Top 10 by Per-Capita Usage')
        ax3.invert_yaxis()

    # 4. GDP vs Per-Capita Usage relationship
    ax4 = axes[1, 1]
    gdp_df = country_df[country_df['variable'] == 'gdp_per_working_age_capita']
    usage_pc_df = country_df[country_df['variable'] == 'usage_per_capita']
    if not gdp_df.empty and not usage_pc_df.empty:
        merged = gdp_df.merge(usage_pc_df, on='geo_id', suffixes=('_gdp', '_usage'))
        merged = merged[merged['value_usage'] > 0]
        # Convert to per million
        ax4.scatter(merged['value_gdp'] / 1000, merged['value_usage'] * 1_000_000, alpha=0.6, c='purple')
        ax4.set_xlabel('GDP per Working-Age Capita ($K)')
        ax4.set_ylabel('Usage per Million People')
        ax4.set_title('GDP vs Per-Capita AI Usage')

    plt.suptitle('Anthropic Economic Index: Country Trends Dashboard\n(Data: August 2025)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('country_trends_dashboard.png', dpi=150)
    plt.close()
    print("Saved: country_trends_dashboard.png")


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("ANTHROPIC ECONOMIC INDEX - COUNTRY TRENDS ANALYSIS")
    print("="*60)

    # Download data from HuggingFace
    df = download_data()

    # Explore dataset structure
    explore_data(df)

    # Country usage analysis
    analyze_country_usage(df)

    # Task patterns
    analyze_task_patterns(df)

    # Collaboration patterns
    analyze_collaboration_patterns(df)

    # Create dashboard
    create_summary_dashboard(df)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated visualizations:")
    print("  - country_usage_count.png")
    print("  - country_per_capita_usage.png")
    print("  - top_tasks.png")
    print("  - automation_vs_augmentation.png")
    print("  - country_trends_dashboard.png")


if __name__ == "__main__":
    main()
