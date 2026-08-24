import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", page_title="Retail Data Explorer")
st.title("🛒 Retail Simulation Data Explorer")

# Tabs for each dataset
tab1, tab2, tab3 = st.tabs([
    "👥 Customers (dim_customers)", 
    "📦 Products (dim_products)", 
    "💳 Transactions (fct_transactions)"
])

with tab1:
    df_customers = pd.read_parquet("data/raw_customers.parquet")
    st.subheader(f"Customers ({len(df_customers):,} rows)")
    st.dataframe(df_customers, use_container_width=True)

with tab2:
    df_products = pd.read_parquet("data/raw_products.parquet")
    st.subheader(f"Products ({len(df_products):,} rows)")
    st.dataframe(df_products, use_container_width=True)

with tab3:
    df_transactions = pd.read_parquet("data/raw_transactions.parquet")
    st.subheader(f"Transactions ({len(df_transactions):,} rows)")
    st.dataframe(df_transactions, use_container_width=True)