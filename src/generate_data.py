import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone

np.random.seed(42)

def generate_v2_retail_dataset(
    n_customers=10000, 
    n_transactions=100000,
    n_products=80
):
    print("--- Generating Realistic Retail Simulation Dataset (V2 with Power Laws) ---")
    now = pd.Timestamp.now(tz=timezone.utc).floor('s')
    
    # -------------------------------------------------------------
    # 1. Product Catalog (dim_products) + Zipf Popularity
    # -------------------------------------------------------------
    print(f"1/3: Building product catalog ({n_products} items)...")
    
    categories_meta = {
        'Apparel': {
            'adjectives': ['Classic', 'Slim-Fit', 'Merino', 'Waterproof', 'Breathable', 'Vintage', 'Organic Cotton', 'Thermal'],
            'nouns': ['Crewneck', 'Denim Jeans', 'Sweater', 'Rain Jacket', 'Shorts', 'Polo Shirt', 'Hoodie', 'Joggers'],
            'price_range': (25.0, 160.0),
            'cost_multiplier': 0.40
        },
        'Electronics': {
            'adjectives': ['Wireless', 'Smart', 'Ultra-HD', 'Noise-Cancelling', 'Portable', 'Ergonomic', 'Fast-Charging', 'Compact'],
            'nouns': ['Headphones', 'Fitness Watch', 'Monitor', 'Keyboard', 'USB-C Hub', 'Earbuds', 'Power Bank', 'Speaker'],
            'price_range': (40.0, 450.0),
            'cost_multiplier': 0.65
        },
        'Home & Kitchen': {
            'adjectives': ['Cast Iron', 'Stainless Steel', 'Ceramic', 'Aromatherapy', 'Insulated', 'Bamboo', 'Non-Stick', 'Artisan'],
            'nouns': ['Dutch Oven', 'Chef Knife', 'Coffee Maker', 'Diffuser', 'Dinnerware Set', 'Blender', 'Cutting Board', 'Toaster'],
            'price_range': (20.0, 250.0),
            'cost_multiplier': 0.50
        },
        'Beauty': {
            'adjectives': ['Hydrating', 'Mineral SPF50', 'Botanical', 'Restorative', 'Exfoliating', 'Revitalizing', 'Nourishing', 'Soothing'],
            'nouns': ['Serum', 'Sunscreen', 'Cleanser', 'Night Cream', 'Face Mask', 'Toner', 'Lip Balm', 'Body Scrub'],
            'price_range': (15.0, 95.0),
            'cost_multiplier': 0.35
        },
        'Sports': {
            'adjectives': ['High-Density', 'Adjustable', 'Insulated', 'Trail', 'Lightweight', 'Heavy-Duty', 'Shock-Absorbing', 'All-Weather'],
            'nouns': ['Yoga Mat', 'Dumbbell Set', 'Hydration Flask', 'Backpack', 'Resistance Bands', 'Jump Rope', 'Gym Bag', 'Foam Roller'],
            'price_range': (20.0, 220.0),
            'cost_multiplier': 0.55
        }
    }
    
    product_rows = []
    prod_counter = 1
    for cat_name, meta in categories_meta.items():
        min_p, max_p = meta['price_range']
        cost_mult = meta['cost_multiplier']
        for adj in meta['adjectives']:
            for noun in meta['nouns'][:2]:
                base_price = round(np.random.uniform(min_p, max_p), 2)
                cost_price = round(base_price * cost_mult, 2)
                product_rows.append({
                    'product_id': f"PROD_{prod_counter:04d}",
                    'product_name': f"{adj} {noun}",
                    'category': cat_name,
                    'base_price': base_price,
                    'cost_price': cost_price
                })
                prod_counter += 1
                
    df_products = pd.DataFrame(product_rows)
    
    # Zipf Distribution for Product Popularity (Top 20% products generate ~80% volume)
    zipf_weights = np.random.zipf(a=1.6, size=len(df_products)).astype(float)
    product_popularity_probs = zipf_weights / zipf_weights.sum()

    # -------------------------------------------------------------
    # 2. Customer Profiles (dim_customers) + Pareto Spending Weights
    # -------------------------------------------------------------
    print(f"2/3: Generating {n_customers:,} customer profiles...")
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
    
    days_ago = np.random.randint(30, 730, size=n_customers)
    signup_dates = now - pd.to_timedelta(days_ago, unit='D')
    
    states = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT']
    state_probs = [0.32, 0.26, 0.20, 0.10, 0.07, 0.03, 0.02]
    
    # Pareto frequency weight: 20% high-frequency buyers, 80% casual/churned
    cust_frequency_weights = np.random.exponential(scale=1.0, size=n_customers).clip(0.05, 6.0)
    cust_frequency_probs = cust_frequency_weights / cust_frequency_weights.sum()
    
    df_customers = pd.DataFrame({
        'customer_id': customer_ids,
        'signup_date': signup_dates,
        'age': np.random.normal(38, 12, n_customers).clip(18, 75).astype(int),
        'gender': np.random.choice(['Female', 'Male', 'Non-Binary', 'Undisclosed'], size=n_customers, p=[0.49, 0.47, 0.02, 0.02]),
        'state': np.random.choice(states, size=n_customers, p=state_probs),
        'acquisition_channel': np.random.choice(['Google Search', 'Paid Social', 'Email Campaign', 'Organic', 'Referral'], size=n_customers, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
        'membership_tier': np.random.choice(['Standard', 'Silver', 'Gold'], size=n_customers, p=[0.60, 0.25, 0.15]),
        'email_opt_in': np.random.binomial(n=1, p=0.72, size=n_customers)
    })

    # -------------------------------------------------------------
    # 3. Transactions with Strict Date Integrity, Seasonality & Power Laws
    # -------------------------------------------------------------
    print(f"3/3: Generating {n_transactions:,} transactions (Power Laws + Strict Integrity)...")
    
    # Select customer by purchase frequency distribution
    cust_idx = np.random.choice(len(df_customers), size=n_transactions, p=cust_frequency_probs)
    selected_customers = df_customers.iloc[cust_idx].reset_index(drop=True)
    
    # Seasonality mixture: 75% uniform, 25% peak clustering
    base_random = np.random.uniform(0.0, 1.0, size=n_transactions)
    seasonal_boost = np.random.beta(a=3, b=1.5, size=n_transactions)
    random_fractions = (0.75 * base_random + 0.25 * seasonal_boost).clip(0.0, 0.99)
    
    # Calculate transaction timestamps (strictly between signup_date and now)
    cust_signups = selected_customers['signup_date'].values
    time_deltas = (now.to_numpy() - cust_signups).astype('timedelta64[s]')
    trans_offsets = (time_deltas.astype('int64') * random_fractions).astype('timedelta64[s]')
    trans_dates = cust_signups + trans_offsets
    
    # Select products by Zipf popularity distribution
    prod_idx = np.random.choice(len(df_products), size=n_transactions, p=product_popularity_probs)
    selected_prods = df_products.iloc[prod_idx].reset_index(drop=True)
    
    quantities = np.random.poisson(lam=1.3, size=n_transactions).clip(1, 6)
    discounts = np.random.choice([0, 5, 10, 15, 20, 30], size=n_transactions, p=[0.55, 0.15, 0.12, 0.10, 0.05, 0.03])
    
    unit_prices = selected_prods['base_price'].values
    net_amounts = np.round((unit_prices * quantities) * (1.0 - (discounts / 100.0)), 2)
    
    channels = np.random.choice(['Web', 'Mobile App', 'In-Store'], size=n_transactions, p=[0.45, 0.40, 0.15])
    shipping_days = np.where(channels == 'In-Store', 0, np.random.poisson(lam=3.5, size=n_transactions).clip(1, 12))
    
    # Returns based on category & shipping delays
    is_apparel = (selected_prods['category'] == 'Apparel').values
    return_prob = (0.04 + 0.08 * is_apparel + 0.05 * (shipping_days > 6)).clip(0, 0.35)
    is_returned = np.random.binomial(n=1, p=return_prob, size=n_transactions)
    
    df_transactions = pd.DataFrame({
        'transaction_id': [f"TXN_{i:08d}" for i in range(1, n_transactions + 1)],
        'customer_id': selected_customers['customer_id'].values,
        'product_id': selected_prods['product_id'].values,
        'transaction_date': pd.to_datetime(trans_dates),
        'quantity': quantities,
        'unit_price': unit_prices,
        'discount_applied_pct': discounts,
        'total_amount': net_amounts,
        'channel': channels,
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Apple Pay', 'Debit Card', 'BNPL'], size=n_transactions, p=[0.40, 0.25, 0.15, 0.12, 0.08]),
        'shipping_days': shipping_days,
        'is_returned': is_returned
    })

    # Save to Parquet
    os.makedirs("data", exist_ok=True)
    df_products.to_parquet("data/raw_products.parquet", index=False)
    df_customers.to_parquet("data/raw_customers.parquet", index=False)
    df_transactions.to_parquet("data/raw_transactions.parquet", index=False)
    
    print("\nDataset Generated Successfully:")
    print(f" - data/raw_products.parquet     : {len(df_products):,} rows")
    print(f" - data/raw_customers.parquet    : {len(df_customers):,} rows")
    print(f" - data/raw_transactions.parquet : {len(df_transactions):,} rows")

if __name__ == "__main__":
    generate_v2_retail_dataset()