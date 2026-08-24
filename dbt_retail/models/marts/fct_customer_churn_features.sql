{{ config(
    materialized='table'
) }}

with customers as (
    select * from {{ ref('stg_customers') }}
),

transactions as (
    select * from {{ ref('stg_transactions') }}
),

products as (
    select * from {{ ref('stg_products') }}
),

-- Reference snapshot timestamp (latest transaction timestamp in data)
snapshot as (
    select max(transaction_at) as max_transaction_at
    from transactions
),

-- Aggregate transaction & product metrics at the customer level
customer_orders as (
    select
        t.customer_id,
        count(t.transaction_id) as total_orders,
        sum(t.quantity) as total_units_purchased,
        sum(t.net_amount) as lifetime_spend,
        avg(t.net_amount) as avg_order_value,
        avg(t.discount_pct) as avg_discount_pct,
        
        -- Discount sensitivity
        safe_divide(countif(t.discount_pct > 0), count(t.transaction_id)) as discount_order_share,
        
        -- Operational friction & returns
        countif(t.is_returned) as total_returns,
        safe_divide(countif(t.is_returned), count(t.transaction_id)) as return_rate,
        avg(t.shipping_days) as avg_shipping_days,
        max(t.shipping_days) as max_shipping_days,
        
        -- Channel breakdown
        safe_divide(countif(t.channel = 'Web'), count(t.transaction_id)) as web_order_share,
        safe_divide(countif(t.channel = 'Mobile App'), count(t.transaction_id)) as app_order_share,
        safe_divide(countif(t.channel = 'In-Store'), count(t.transaction_id)) as store_order_share,
        
        -- Product category diversity
        count(distinct p.category) as distinct_categories_bought,
        countif(p.category = 'Apparel') as apparel_item_count,
        countif(p.category = 'Electronics') as electronics_item_count,
        
        -- Recency calculation
        max(t.transaction_at) as last_order_at
    from transactions t
    left join products p
        on t.product_id = p.product_id
    group by 1
)

select
    c.customer_id,
    
    -- 1. Demographic & Profile Features
    c.age,
    c.gender,
    c.state,
    c.acquisition_channel,
    c.membership_tier,
    c.is_email_opted_in,
    
    -- Customer tenure in days (from signup to snapshot)
    date_diff(date(s.max_transaction_at), date(c.signup_at), day) as tenure_days,
    
    -- 2. RFM Features
    coalesce(date_diff(date(s.max_transaction_at), date(co.last_order_at), day), 999) as recency_days,
    coalesce(co.total_orders, 0) as frequency,
    coalesce(round(co.lifetime_spend, 2), 0.0) as monetary_value,
    coalesce(round(co.avg_order_value, 2), 0.0) as avg_order_value,
    coalesce(co.total_units_purchased, 0) as total_units_purchased,
    
    -- 3. Behavioral & Pricing Features
    coalesce(round(co.avg_discount_pct, 2), 0.0) as avg_discount_pct,
    coalesce(round(co.discount_order_share, 4), 0.0) as discount_order_share,
    
    -- 4. Operational Friction Features (Key for SHAP Explainability)
    coalesce(co.total_returns, 0) as total_returns,
    coalesce(round(co.return_rate, 4), 0.0) as return_rate,
    coalesce(round(co.avg_shipping_days, 2), 0.0) as avg_shipping_days,
    coalesce(co.max_shipping_days, 0) as max_shipping_days,
    
    -- 5. Channel Distribution
    coalesce(round(co.web_order_share, 4), 0.0) as web_order_share,
    coalesce(round(co.app_order_share, 4), 0.0) as app_order_share,
    coalesce(round(co.store_order_share, 4), 0.0) as store_order_share,
    
    -- 6. Product Category Affinity
    coalesce(co.distinct_categories_bought, 0) as distinct_categories_bought,
    coalesce(co.apparel_item_count, 0) as apparel_item_count,
    coalesce(co.electronics_item_count, 0) as electronics_item_count,
    
    -- 7. Target Variable: Churn Label (1 if inactive for > 90 days, else 0)
    case 
        when coalesce(date_diff(date(s.max_transaction_at), date(co.last_order_at), day), 999) > 90 then 1
        else 0 
    end as is_churned

from customers c
cross join snapshot s
left join customer_orders co
    on c.customer_id = co.customer_id