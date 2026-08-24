import os
import pandas as pd
from google.cloud import bigquery

KEY_PATH = "gcp_key.json"

def verify():
    print("=" * 70)
    print("VALIDATING BIGQUERY FEATURE MART: fct_customer_churn_features")
    print("=" * 70)
    
    client = bigquery.Client.from_service_account_json(KEY_PATH)
    project_id = client.project
    
    # 1. Query the Mart Table from BigQuery
    query = f"""
        SELECT *
        FROM `{project_id}.retail_analytics.fct_customer_churn_features`
    """
    print("Fetching feature table from BigQuery...")
    df = client.query(query).to_dataframe()
    
    print(f"Table Loaded: {df.shape[0]:,} rows (customers), {df.shape} columns.\n")
    
    # -------------------------------------------------------------
    # 1. Pareto Principle Verification (80/20 Spend Share)
    # -------------------------------------------------------------
    total_revenue = df['monetary_value'].sum()
    top_20_pct_count = int(len(df) * 0.20)
    top_20_revenue = df['monetary_value'].nlargest(top_20_pct_count).sum()
    pareto_share = (top_20_revenue / total_revenue) * 100
    
    p80_threshold = df['monetary_value'].quantile(0.80)
    p20_threshold = df['monetary_value'].quantile(0.20)
    
    print("--- 1. Pareto / Power-Law Distribution Check ---")
    print(f"Total Revenue Generated      : ${total_revenue:,.2f}")
    print(f"Top 20% Spenders Revenue     : ${top_20_revenue:,.2f} ({pareto_share:.1f}% of total)")
    print(f"80th Percentile Cutoff (P80) : ${p80_threshold:,.2f} (Top 20% spent at least this amount)")
    print(f"20th Percentile Cutoff (P20) : ${p20_threshold:,.2f}")
    
    if pareto_share >= 60.0:
        print(">> PASS: Strong Power-Law / Pareto distribution confirmed in BigQuery!\n")
    else:
        print(">> Note: Distribution is more uniform than expected.\n")
        
    # -------------------------------------------------------------
    # 2. Target Variable & Class Balance Check
    # -------------------------------------------------------------
    churn_counts = df['is_churned'].value_counts()
    churn_rate = df['is_churned'].mean() * 100
    print("--- 2. Churn Target Class Balance ---")
    print(f"Active Customers (0) : {churn_counts.get(0, 0):,} ({100 - churn_rate:.1f}%)")
    print(f"Churned Customers (1): {churn_counts.get(1, 0):,} ({churn_rate:.1f}%)")
    print(f"Overall Churn Rate   : {churn_rate:.1f}%\n")
    
    # -------------------------------------------------------------
    # 3. Behavioral Friction vs. Churn Check (Sanity Check)
    # -------------------------------------------------------------
    print("--- 3. Operational Friction Signals by Churn Status ---")
    friction_summary = df.groupby('is_churned').agg(
        avg_orders=('frequency', 'mean'),
        avg_spend=('monetary_value', 'mean'),
        avg_returns=('total_returns', 'mean'),
        avg_shipping_delay=('avg_shipping_days', 'mean'),
        avg_discount_share=('discount_order_share', 'mean')
    ).round(2)
    friction_summary.index = ['Active (0)', 'Churned (1)']
    print(friction_summary.to_string())
    print("=" * 70)

if __name__ == "__main__":
    verify()