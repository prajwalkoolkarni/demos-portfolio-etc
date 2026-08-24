import os
import json
from google.cloud import bigquery

KEY_PATH = "gcp_key.json"
DATASET_ID = "retail_raw"
LOCATION = "australia-southeast1"  # Uses the Sydney/Brisbane regional data center

def upload():
    if not os.path.exists(KEY_PATH):
        print(f"Error: Could not find '{KEY_PATH}' in your project root folder.")
        print("Please make sure you moved the downloaded key to the project root and renamed it to gcp_key.json")
        return

    # Authenticate directly using the Service Account JSON key
    print("1/4: Authenticating with Google Cloud using gcp_key.json...")
    client = bigquery.Client.from_service_account_json(KEY_PATH)
    project_id = client.project
    print(f"Connected to GCP Project: {project_id}")

    # Create the dataset in BigQuery if it doesn't already exist
    print(f"2/4: Ensuring dataset '{DATASET_ID}' exists...")
    dataset_ref = bigquery.DatasetReference(project_id, DATASET_ID)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = LOCATION
    dataset = client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset '{DATASET_ID}' is ready in location '{LOCATION}'.")

    # Define the 3 Parquet files to upload
    tables_to_upload = {
        'products': 'data/raw_products.parquet',
        'customers': 'data/raw_customers.parquet',
        'transactions': 'data/raw_transactions.parquet'
    }

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        source_format=bigquery.SourceFormat.PARQUET
    )

    print("3/4: Uploading Parquet tables to BigQuery...")
    for table_name, file_path in tables_to_upload.items():
        if not os.path.exists(file_path):
            print(f"File {file_path} not found! Run 'python src/generate_data.py' first.")
            continue
            
        print(f" -> Uploading {file_path} into table '{DATASET_ID}.{table_name}'...")
        table_ref = dataset_ref.table(table_name)
        
        with open(file_path, "rb") as f:
            load_job = client.load_table_from_file(f, table_ref, job_config=job_config)
            
        load_job.result()  # Wait for BigQuery to finish ingestion
        table = client.get_table(table_ref)
        print(f"    Loaded {table.num_rows:,} rows into {DATASET_ID}.{table_name}")

    print("\n4/4: Success! All 3 tables are now live in BigQuery.")

if __name__ == "__main__":
    upload()