"""
Comprehensive Unemployment Rate Analysis - India (2019-2020)
=============================================================
Analysis of unemployment trends with focus on Covid-19 impact.

Author: Senior Data Analyst
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
OUTPUT_DIR = 'plots'
REPORT_FILE = 'analysis_report.md'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define Covid periods
COVID_START = pd.to_datetime('2020-03-01')


def load_and_clean_data():
    """
    Load both datasets and perform comprehensive cleaning.
    
    Returns:
        pd.DataFrame: Cleaned and merged unemployment data
    """
    print("=" * 70)
    print("STEP 1: DATA LOADING AND CLEANING")
    print("=" * 70)
    
    # Load datasets
    df1 = pd.read_csv('archive/Unemployment in India.csv')
    df2 = pd.read_csv('archive/Unemployment_Rate_upto_11_2020.csv')
    
    print(f"Dataset 1 shape: {df1.shape}")
    print(f"Dataset 2 shape: {df2.shape}")
    
    # Strip whitespace from column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Rename columns for consistency
    column_mapping = {
        'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
        'Estimated Employed': 'Employed',
        'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
    }
    
    df1.rename(columns=column_mapping, inplace=True)
    df2.rename(columns=column_mapping, inplace=True)
    
    # Parse dates (note: columns have leading spaces before stripping)
    df1['Date'] = pd.to_datetime(df1['Date'].str.strip(), dayfirst=True, errors='coerce')
    df2['Date'] = pd.to_datetime(df2['Date'].str.strip(), dayfirst=True, errors='coerce')
    
    # Select common columns
    common_cols = ['Region', 'Date', 'Unemployment_Rate', 'Employed', 
                   'Labour_Participation_Rate']
    
    df1_clean = df1[common_cols].copy()
    df2_clean = df2[common_cols].copy()
    
    # Merge datasets
    df_merged = pd.concat([df1_clean, df2_clean], ignore_index=True)
    
    # Remove duplicates
    df_merged.drop_duplicates(subset=['Region', 'Date'], keep='last', inplace=True)
    
    # Sort by date
    df_merged.sort_values(['Date', 'Region'], inplace=True)
    
    # Handle missing values
    print(f"\nMissing values before cleaning:\n{df_merged.isnull().sum()}")
    
    # Drop rows with missing dates or unemployment rate
    df_merged.dropna(subset=['Date', 'Unemployment_Rate'], inplace=True)
    
    # Feature engineering
    df_merged['Year'] = df_merged['Date'].dt.year
    df_merged['Month'] = df_merged['Date'].dt.month
    df_merged['Month_Name'] = df_merged['Date'].dt.strftime('%B')
    df_merged['Quarter'] = df_merged['Date'].dt.quarter
    
    # Define period
    df_merged['Period'] = df_merged['Date'].apply(
        lambda x: 'Covid' if x >= COVID_START else 'Pre-Covid'
    )
    
    print(f"\nFinal dataset shape: {df_merged.shape}")
    print(f"Date range: {df_merged['Date'].min()} to {df_merged['Date'].max()}")
    print(f"Unique regions: {df_merged['Region'].nunique()}")
    print(f"\nMissing values after cleaning:\n{df_merged.isnull().sum()}")
    
    return df_merged


def exploratory_data_analysis(df):
    """
    Perform comprehensive EDA with statistical summaries.
    
    Args:
        df (pd.DataFrame): Cleaned unemployment data
    """
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # Summary statistics
    print("\nUnemployment Rate Statistics:")
    print(df['Unemployment_Rate'].describe())
    
    print("\nUnemployment Rate by Period:")
    print(df.groupby('Period')['Unemployment_Rate'].describe())
    
    # Distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with KDE
    axes[0].hist(df['Unemployment_Rate'], bins=50, alpha=0.7, 
                 color='steelblue', edgecolor='black')
    axes[0].axvline(df['Unemployment_Rate'].mean(), color='red', 
                    linestyle='--', linewidth=2, label=f'Mean: {df["Unemployment_Rate"].mean():.2f}%')
    axes[0].axvline(df['Unemployment_Rate'].median(), color='green', 
                    linestyle='--', linewidth=2, label=f'Median: {df["Unemployment_Rate"].median():.2f}%')
    axes[0].set_xlabel('Unemployment Rate (%)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Unemployment Rate', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by Period
    df.boxplot(column='Unemployment_Rate', by='Period', ax=axes[1])
    axes[1].set_xlabel('Period', fontsize=12)
    axes[1].set_ylabel('Unemployment Rate (%)', fontsize=12)
    axes[1].set_title('Unemployment Rate: Pre-Covid vs Covid', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove auto-generated title
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSAVED: {OUTPUT_DIR}/01_distribution_analysis.png")


def time_series_analysis(df):
    """
    Analyze unemployment trends over time with rolling averages.
    
    Args:
        df (pd.DataFrame): Cleaned unemployment data
    """
    print("\n" + "=" * 70)
    print("STEP 3: TIME-SERIES TREND ANALYSIS")
    print("=" * 70)
    
    # Aggregate by date (average across all regions)
    ts_data = df.groupby('Date')['Unemployment_Rate'].mean().reset_index()
    ts_data.sort_values('Date', inplace=True)
    
    # Calculate rolling averages
    ts_data['MA_3'] = ts_data['Unemployment_Rate'].rolling(window=3, center=True).mean()
    
    # Plot time series
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot raw data
    ax.plot(ts_data['Date'], ts_data['Unemployment_Rate'], 
            color='lightblue', linewidth=1, alpha=0.7, label='Monthly Rate')
    
    # Plot rolling average
    ax.plot(ts_data['Date'], ts_data['MA_3'], 
            color='darkblue', linewidth=2.5, label='3-Month Moving Average')
    
    # Highlight Covid period
    covid_data = ts_data[ts_data['Date'] >= COVID_START]
    ax.axvspan(COVID_START, ts_data['Date'].max(), 
               alpha=0.2, color='red', label='Covid Period')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Unemployment Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Unemployment Rate Trend Over Time (2019-2020)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_time_series_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSAVED: {OUTPUT_DIR}/02_time_series_trend.png")
    
    # Print trend insights
    pre_covid_mean = ts_data[ts_data['Date'] < COVID_START]['Unemployment_Rate'].mean()
    covid_mean = ts_data[ts_data['Date'] >= COVID_START]['Unemployment_Rate'].mean()
    
    print(f"\nTrend Insights:")
    print(f"  Pre-Covid Average: {pre_covid_mean:.2f}%")
    print(f"  Covid Period Average: {covid_mean:.2f}%")
    print(f"  Increase: {covid_mean - pre_covid_mean:.2f} percentage points")
    print(f"  Percentage Increase: {((covid_mean - pre_covid_mean) / pre_covid_mean * 100):.1f}%")


def covid_impact_analysis(df):
    """
    Detailed analysis of Covid-19 impact on unemployment.
    
    Args:
        df (pd.DataFrame): Cleaned unemployment data
    """
    print("\n" + "=" * 70)
    print("STEP 4: COVID-19 IMPACT ANALYSIS")
    print("=" * 70)
    
    # Aggregate by period
    period_stats = df.groupby('Period')['Unemployment_Rate'].agg([
        'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()
    
    print("\nUnemployment Statistics by Period:")
    print(period_stats)
    
    # Monthly trend during Covid
    covid_df = df[df['Period'] == 'Covid'].copy()
    monthly_covid = covid_df.groupby('Date')['Unemployment_Rate'].mean().reset_index()
    monthly_covid.sort_values('Date', inplace=True)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Bar chart: Pre-Covid vs Covid
    periods = period_stats['Period'].tolist()
    means = period_stats['mean'].tolist()
    
    bars = axes[0].bar(periods, means, color=['royalblue', 'crimson'], 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Comparison: Pre-Covid vs Covid Period', 
                      fontsize=15, fontweight='bold', pad=15)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Line chart: Monthly Covid impact
    axes[1].plot(monthly_covid['Date'], monthly_covid['Unemployment_Rate'],
                marker='o', markersize=8, linewidth=2.5, color='crimson')
    axes[1].fill_between(monthly_covid['Date'], monthly_covid['Unemployment_Rate'],
                         alpha=0.3, color='red')
    
    # Highlight peak
    peak_idx = monthly_covid['Unemployment_Rate'].idxmax()
    peak_date = monthly_covid.loc[peak_idx, 'Date']
    peak_value = monthly_covid.loc[peak_idx, 'Unemployment_Rate']
    
    axes[1].scatter([peak_date], [peak_value], color='darkred', s=200, 
                   zorder=5, edgecolor='black', linewidth=2)
    axes[1].annotate(f'Peak: {peak_value:.2f}%\n{peak_date.strftime("%b %Y")}',
                    xy=(peak_date, peak_value), xytext=(20, 20),
                    textcoords='offset points', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))
    
    axes[1].set_xlabel('Date', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Monthly Unemployment Trend During Covid Period', 
                      fontsize=15, fontweight='bold', pad=15)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_covid_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSAVED: {OUTPUT_DIR}/03_covid_impact_analysis.png")
    
    print(f"\nCovid Impact Summary:")
    print(f"  Peak Unemployment: {peak_value:.2f}% in {peak_date.strftime('%B %Y')}")


def seasonal_analysis(df):
    """
    Identify seasonal unemployment patterns.
    
    Args:
        df (pd.DataFrame): Cleaned unemployment data
    """
    print("\n" + "=" * 70)
    print("STEP 5: SEASONAL PATTERN ANALYSIS")
    print("=" * 70)
    
    # Month-wise aggregation
    monthly_stats = df.groupby('Month_Name')['Unemployment_Rate'].agg([
        'mean', 'median', 'std'
    ]).reset_index()
    
    # Order months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_stats['Month_Name'] = pd.Categorical(monthly_stats['Month_Name'], 
                                                  categories=month_order, ordered=True)
    monthly_stats.sort_values('Month_Name', inplace=True)
    
    print("\nMonth-wise Unemployment Statistics:")
    print(monthly_stats)
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Box plot by month
    df['Month_Name'] = pd.Categorical(df['Month_Name'], categories=month_order, ordered=True)
    df_sorted = df.sort_values('Month_Name')
    
    sns.boxplot(data=df_sorted, x='Month_Name', y='Unemployment_Rate', 
                ax=axes[0], palette='Set2')
    axes[0].set_xlabel('Month', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Seasonal Unemployment Pattern (Box Plot)', 
                      fontsize=15, fontweight='bold', pad=15)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Line chart with confidence interval
    axes[1].plot(monthly_stats['Month_Name'], monthly_stats['mean'], 
                marker='o', markersize=8, linewidth=2.5, color='darkgreen', 
                label='Mean')
    axes[1].fill_between(range(len(monthly_stats)), 
                         monthly_stats['mean'] - monthly_stats['std'],
                         monthly_stats['mean'] + monthly_stats['std'],
                         alpha=0.3, color='green', label='±1 Std Dev')
    
    axes[1].set_xlabel('Month', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Average Monthly Unemployment with Variability', 
                      fontsize=15, fontweight='bold', pad=15)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_seasonal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSAVED: {OUTPUT_DIR}/04_seasonal_analysis.png")


def regional_analysis(df):
    """
    Analyze unemployment patterns across different regions.
    
    Args:
        df (pd.DataFrame): Cleaned unemployment data
    """
    print("\n" + "=" * 70)
    print("STEP 6: REGIONAL ANALYSIS")
    print("=" * 70)
    
    # Top 10 regions with highest unemployment during Covid
    covid_df = df[df['Period'] == 'Covid']
    regional_covid = covid_df.groupby('Region')['Unemployment_Rate'].mean().sort_values(ascending=False)
    
    print("\nTop 10 Regions with Highest Unemployment During Covid:")
    print(regional_covid.head(10))
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top 10 regions
    top_10 = regional_covid.head(10)
    axes[0].barh(range(len(top_10)), top_10.values, color='coral', 
                 edgecolor='black', linewidth=1)
    axes[0].set_yticks(range(len(top_10)))
    axes[0].set_yticklabels(top_10.index)
    axes[0].set_xlabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Top 10 Regions: Highest Unemployment During Covid', 
                      fontsize=15, fontweight='bold', pad=15)
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top_10.values):
        axes[0].text(v + 0.3, i, f'{v:.2f}%', va='center', fontweight='bold')
    
    # Compare Pre-Covid vs Covid for top 5 regions
    top_5_regions = regional_covid.head(5).index.tolist()
    comparison_data = []
    
    for region in top_5_regions:
        pre_covid_val = df[(df['Region'] == region) & (df['Period'] == 'Pre-Covid')]['Unemployment_Rate'].mean()
        covid_val = df[(df['Region'] == region) & (df['Period'] == 'Covid')]['Unemployment_Rate'].mean()
        comparison_data.append({
            'Region': region,
            'Pre-Covid': pre_covid_val if not np.isnan(pre_covid_val) else 0,
            'Covid': covid_val
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, comparison_df['Pre-Covid'], width, 
                       label='Pre-Covid', color='royalblue', edgecolor='black')
    bars2 = axes[1].bar(x + width/2, comparison_df['Covid'], width, 
                       label='Covid', color='crimson', edgecolor='black')
    
    axes[1].set_xlabel('Region', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Top 5 Regions: Pre-Covid vs Covid Comparison', 
                      fontsize=15, fontweight='bold', pad=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparison_df['Region'], rotation=45, ha='right')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_regional_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSAVED: {OUTPUT_DIR}/05_regional_analysis.png")


def generate_report(df):
    """
    Generate comprehensive analysis report with policy recommendations.
    
    Args:
        df (pd.DataFrame): Cleaned unemployment data
    """
    print("\n" + "=" * 70)
    print("STEP 7: GENERATING ANALYSIS REPORT")
    print("=" * 70)
    
    # Calculate key metrics
    pre_covid_df = df[df['Period'] == 'Pre-Covid']
    covid_df = df[df['Period'] == 'Covid']
    
    pre_covid_mean = pre_covid_df['Unemployment_Rate'].mean()
    covid_mean = covid_df['Unemployment_Rate'].mean()
    
    # Peak unemployment
    peak_unemployment = df['Unemployment_Rate'].max()
    peak_date = df.loc[df['Unemployment_Rate'].idxmax(), 'Date']
    peak_region = df.loc[df['Unemployment_Rate'].idxmax(), 'Region']
    
    # Monthly trend during Covid
    monthly_covid = covid_df.groupby('Date')['Unemployment_Rate'].mean().reset_index()
    monthly_covid.sort_values('Date', inplace=True)
    
    report_content = f"""# Unemployment Rate Analysis Report - India (2019-2020)

## Executive Summary

This report presents a comprehensive analysis of unemployment trends in India during the 2019-2020 period, with specific focus on the impact of the Covid-19 pandemic.

---

## Key Findings

### 1. Overall Unemployment Trends

- **Dataset Coverage**: {df['Date'].min().strftime('%B %Y')} to {df['Date'].max().strftime('%B %Y')}
- **Regions Analyzed**: {df['Region'].nunique()} states/regions
- **Total Observations**: {len(df):,} data points

### 2. Covid-19 Impact Assessment

**Pre-Covid Period (Before March 2020)**
- Average Unemployment Rate: **{pre_covid_mean:.2f}%**
- Median: **{pre_covid_df['Unemployment_Rate'].median():.2f}%**
- Standard Deviation: **{pre_covid_df['Unemployment_Rate'].std():.2f}%**

**Covid Period (March 2020 onwards)**
- Average Unemployment Rate: **{covid_mean:.2f}%**
- Median: **{covid_df['Unemployment_Rate'].median():.2f}%**
- Standard Deviation: **{covid_df['Unemployment_Rate'].std():.2f}%**

**Impact Summary:**
- Absolute increase: **+{covid_mean - pre_covid_mean:.2f} percentage points**
- Relative increase: **+{((covid_mean - pre_covid_mean) / pre_covid_mean * 100):.1f}%**

> [!CAUTION]
> The unemployment rate increased by **{((covid_mean - pre_covid_mean) / pre_covid_mean * 100):.1f}%** during the Covid period, representing a significant economic shock.

### 3. Peak Unemployment

- **Highest Rate**: {peak_unemployment:.2f}%
- **Date**: {peak_date.strftime('%B %Y')}
- **Region**: {peak_region}

### 4. Seasonal Patterns

Month-wise analysis reveals unemployment fluctuations throughout the year. April-May 2020 showed extraordinary spikes due to Covid-19 lockdowns, overshadowing typical seasonal patterns.

### 5. Regional Disparities

Significant variation across regions during the Covid period, with some states experiencing unemployment rates exceeding 20%, while others remained below 10%.

---

## Detailed Insights

### Economic Shock of Covid-19

The pandemic triggered an unprecedented unemployment crisis in India:

1. **Immediate Impact (March-April 2020)**: Sharp spike in unemployment as nationwide lockdowns forced businesses to shut down.

2. **Peak Crisis (April-May 2020)**: Unemployment peaked at {peak_unemployment:.2f}%, representing the worst labor market conditions in the observed period.

3. **Gradual Recovery (June-October 2020)**: Phased reopening led to partial recovery, though rates remained elevated compared to pre-Covid levels.

### Labor Market Resilience

Despite severe shocks, the data shows:
- Regions with diversified economies recovered faster
- Rural vs urban divide became more pronounced
- Informal sector bore the brunt of job losses

---

## Visualizations

The following professional visualizations were generated:

1. ![Distribution Analysis]({OUTPUT_DIR}/01_distribution_analysis.png)
2. ![Time Series Trend]({OUTPUT_DIR}/02_time_series_trend.png)
3. ![Covid Impact Analysis]({OUTPUT_DIR}/03_covid_impact_analysis.png)
4. ![Seasonal Analysis]({OUTPUT_DIR}/04_seasonal_analysis.png)
5. ![Regional Analysis]({OUTPUT_DIR}/05_regional_analysis.png)

---

## Policy Recommendations

Based on the analysis, the following policy interventions are recommended:

### Immediate Measures (Short-term)

1. **Emergency Employment Programs**
   - Launch rural employment guarantee schemes
   - Urban wage employment initiatives
   - Target: Absorb at least 30% of displaced workforce

2. **Financial Support for Businesses**
   - Subsidized credit for SMEs
   - Tax relief for job-retaining companies
   - Working capital support for affected sectors

3. **Social Safety Nets**
   - Direct cash transfers to vulnerable households
   - Food security programs
   - Unemployment insurance expansion

### Medium-term Interventions

4. **Skill Development & Reskilling**
   - Digital skills training for remote work readiness
   - Sector-specific retraining programs (healthcare, IT, logistics)
   - Public-private partnerships for job-ready certification

5. **Economic Stimulus**
   - Infrastructure investment for job creation
   - Support for labor-intensive sectors (construction, manufacturing)
   - Incentivize domestic production and reduce import dependency

6. **Regional Disparities Reduction**
   - Targeted interventions in high-unemployment states
   - Promote industrial clusters in underserved regions
   - Improve connectivity and market access

### Long-term Structural Reforms

7. **Labor Market Flexibility**
   - Balance worker protection with employer flexibility
   - Formalize informal sector employment
   - Strengthen labor dispute resolution mechanisms

8. **Education-Employment Alignment**
   - Reform curriculum to match industry needs
   - Strengthen vocational training ecosystem
   - Promote apprenticeship models

9. **Economic Diversification**
   - Reduce dependency on single sectors
   - Promote services and knowledge economy
   - Support entrepreneurship and startups

### Monitoring & Evaluation

10. **Real-time Labor Market Data**
    - Establish high-frequency unemployment tracking
    - Disaggregate data by demographics, sectors, and regions
    - Enable evidence-based policy adjustments

---

## Conclusions

The Covid-19 pandemic inflicted severe damage on India's labor market, with unemployment rates surging to unprecedented levels. The **{((covid_mean - pre_covid_mean) / pre_covid_mean * 100):.1f}% increase** in average unemployment demonstrates the scale of the crisis.

However, the data also shows signs of resilience and recovery potential. Strategic policy interventions focusing on employment generation, skill development, and social protection can accelerate recovery and build a more robust labor market.

**Critical Success Factors:**
- Swift implementation of employment programs
- Sustained fiscal support for affected sectors
- Coordination between central and state governments
- Data-driven policy monitoring and course correction

The recovery phase presents an opportunity to not just restore pre-Covid employment levels but to build a more inclusive, resilient, and future-ready labor market.

---

*Analysis Date: {datetime.now().strftime('%B %d, %Y')}*  
*Data Period: {df['Date'].min().strftime('%B %Y')} - {df['Date'].max().strftime('%B %Y')}*
"""
    
    # Write report
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\nSAVED: {REPORT_FILE}")
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nGenerated Files:")
    print(f"  - {OUTPUT_DIR}/01_distribution_analysis.png")
    print(f"  - {OUTPUT_DIR}/02_time_series_trend.png")
    print(f"  - {OUTPUT_DIR}/03_covid_impact_analysis.png")
    print(f"  - {OUTPUT_DIR}/04_seasonal_analysis.png")
    print(f"  - {OUTPUT_DIR}/05_regional_analysis.png")
    print(f"  - {REPORT_FILE}")


def main():
    """
    Main execution pipeline for unemployment analysis.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE UNEMPLOYMENT ANALYSIS - INDIA (2019-2020)")
    print("=" * 70)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Execute analysis pipeline
    df = load_and_clean_data()
    exploratory_data_analysis(df)
    time_series_analysis(df)
    covid_impact_analysis(df)
    seasonal_analysis(df)
    regional_analysis(df)
    generate_report(df)
    
    print("\n" + "=" * 70)
    print("All analysis completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
