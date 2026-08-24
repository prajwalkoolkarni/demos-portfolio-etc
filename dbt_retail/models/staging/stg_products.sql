with source as (
    select * from {{ source('retail_raw', 'products') }}
)

select
    product_id,
    product_name,
    category,
    cast(base_price as float64) as base_price,
    cast(cost_price as float64) as cost_price,
    round(cast(base_price as float64) - cast(cost_price as float64), 2) as unit_margin
from source