with source as (
    select * from {{ source('retail_raw', 'transactions') }}
)

select
    transaction_id,
    customer_id,
    product_id,
    case 
        when cast(transaction_date as int64) > 100000000000000000 then timestamp_micros(div(cast(transaction_date as int64), 1000))
        when cast(transaction_date as int64) > 100000000000000 then timestamp_micros(cast(transaction_date as int64))
        when cast(transaction_date as int64) > 100000000000 then timestamp_millis(cast(transaction_date as int64))
        else timestamp_seconds(cast(transaction_date as int64))
    end as transaction_at,
    cast(quantity as int64) as quantity,
    cast(unit_price as float64) as unit_price,
    cast(discount_applied_pct as float64) as discount_pct,
    cast(total_amount as float64) as net_amount,
    channel,
    payment_method,
    cast(shipping_days as int64) as shipping_days,
    cast(is_returned as boolean) as is_returned
from source