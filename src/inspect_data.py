import pandas as pd
import numpy as np

def inspect():
    print("=" * 70)
    print("1. LOADING DATASETS")
    print("=" * 70)
    df_products = pd.read_parquet("data/raw_products.parquet")
    df_customers = pd.read_parquet("data/raw_customers.parquet")
    df_transactions = pd.read_parquet("data/raw_transactions.parquet")
    
    print(f"Products shape     : {df_products.shape} (rows, columns)")
    print(f"Customers shape    : {df_customers.shape}")
    print(f"Transactions shape : {df_transactions.shape}\n")
    
    # -------------------------------------------------------------
    # Sample Rows
    # -------------------------------------------------------------
    print("=" * 70)
    print("2. SAMPLE ROWS (HEAD 3)")
    print("=" * 70)
    print("\n--- Products (dim_products) ---")
    print(df_products.head(3).to_string(index=False))
    
    print("\n--- Customers (dim_customers) ---")
    print(df_customers.head(3).to_string(index=False))
    
    print("\n--- Transactions (fct_transactions) ---")
    print(df_transactions.head(3).to_string(index=False))
    
    # -------------------------------------------------------------
    # Sanity Checks
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. SANITY CHECKS & POWER-LAW DISTRIBUTIONS")
    print("=" * 70)
    
    # Check 1: Date Integrity (Transactions must be on/after signup)
    merged = df_transactions[['transaction_id', 'customer_id', 'transaction_date']].merge(
        df_customers[['customer_id', 'signup_date']], on='customer_id', how='left'
    )
    invalid_dates = (merged['transaction_date'] < merged['signup_date']).sum()
    print(f"Date Integrity Check : {invalid_dates} transactions before customer signup (Must be 0).")
    
    # Check 2: Null values check
    total_nulls = df_products.isnull().sum().sum() + df_customers.isnull().sum().sum() + df_transactions.isnull().sum().sum()
    print(f"Missing Values Check : {total_nulls} null values across all tables.")
    
    # Check 3: Top 5 Hero Products (Zipf law check)
    print("\n--- Top 5 Best-Selling Products (Volume) ---")
    prod_sales = df_transactions.groupby('product_id')['quantity'].sum().reset_index()
    prod_sales = prod_sales.merge(df_products[['product_id', 'product_name', 'category']], on='product_id')
    top_5 = prod_sales.sort_values(by='quantity', ascending=False).head(5)
    print(top_5.to_string(index=False))
    
    # Check 4: Customer Pareto Distribution (Top 20% vs Bottom 80%)
    cust_spend = df_transactions.groupby('customer_id')['total_amount'].sum()
    top_20_pct_count = int(len(cust_spend) * 0.20)
    top_20_spend = cust_spend.nlargest(top_20_pct_count).sum()
    total_spend = cust_spend.sum()
    pct_revenue_from_top_20 = (top_20_spend / total_spend) * 100
    print(f"\nPareto Principle Check: Top 20% of buyers generate {pct_revenue_from_top_20:.1f}% of total revenue.")
    
    # Check 5: Category & Return Breakdown
    print("\n--- Returns & Net Revenue by Category ---")
    tx_prod = df_transactions.merge(df_products[['product_id', 'category']], on='product_id')
    cat_summary = tx_prod.groupby('category').agg(
        orders=('transaction_id', 'count'),
        total_revenue=('total_amount', 'sum'),
        return_rate=('is_returned', 'mean')
    ).reset_index()
    cat_summary['return_rate'] = (cat_summary['return_rate'] * 100).round(2).astype(str) + '%'
    cat_summary['total_revenue'] = cat_summary['total_revenue'].map('${:,.2f}'.format)
    print(cat_summary.to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    inspect()