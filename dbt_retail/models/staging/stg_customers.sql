with source as (
    select * from {{ source('retail_raw', 'customers') }}
)

select
    customer_id,
    case 
        when cast(signup_date as int64) > 100000000000000000 then timestamp_micros(div(cast(signup_date as int64), 1000))
        when cast(signup_date as int64) > 100000000000000 then timestamp_micros(cast(signup_date as int64))
        when cast(signup_date as int64) > 100000000000 then timestamp_millis(cast(signup_date as int64))
        else timestamp_seconds(cast(signup_date as int64))
    end as signup_at,
    cast(age as int64) as age,
    gender,
    state,
    acquisition_channel,
    membership_tier,
    cast(email_opt_in as boolean) as is_email_opted_in
from source