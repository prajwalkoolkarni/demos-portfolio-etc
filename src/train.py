import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import shap

# --- ADD THIS LINE ---
mlflow.set_tracking_uri('sqlite:///mlflow.db')
# ----------------------

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def get_bigquery_client():
    """Environment-aware BigQuery client initialization."""
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "gcp_key.json")
    if os.path.exists(key_path):
        return bigquery.Client.from_service_account_json(key_path)
    return bigquery.Client()

def train_model():
    print("=" * 70)
    print("STEP 6: XGBOOST TRAINING, PIPELINE PACKAGING & MLFLOW TRACKING")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # 1. Fetch Features from BigQuery
    # -------------------------------------------------------------
    print("1/5: Querying feature mart from BigQuery...")
    client = get_bigquery_client()
    query = f"""
        SELECT *
        FROM `{client.project}.retail_analytics.fct_customer_churn_features`
    """
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df):,} customer records with {df.shape} columns.")
    
    # Check for missing values
    null_count = df.isnull().sum().sum()
    print(f"Data Quality Check: {null_count} missing values detected.")
    
    # -------------------------------------------------------------
    # 2. Preprocessing & Anti-Leakage Separation
    # -------------------------------------------------------------
    print("2/5: Preparing feature matrices (preventing target leakage)...")
    
    y = df['is_churned'].astype(int)
    
    # Drop identifiers and target-leakage columns (recency_days)
    drop_cols = ['customer_id', 'recency_days', 'is_churned']
    X = df.drop(columns=drop_cols)
    
    categorical_cols = ['gender', 'state', 'acquisition_channel', 'membership_tier']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    
    print(f" -> Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f" -> Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Save baseline reference dataset for drift monitoring
    df.to_parquet(os.path.join(ARTIFACTS_DIR, "reference_data.parquet"), index=False)
    
    # Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Column Transformer for One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # -------------------------------------------------------------
    # 3. Model Architecture & Hyperparameters
    # -------------------------------------------------------------
    print("3/5: Fitting Unified End-to-End Pipeline...")
    
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    
    model_params = {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'logloss',
        'enable_categorical': False  # <-- ADD THIS LINE
    }
    
    # Unified Production Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**model_params))
    ])
    
    full_pipeline.fit(X_train, y_train)
    
    # -------------------------------------------------------------
    # 4. Evaluation Metrics on Held-Out Test Set
    # -------------------------------------------------------------
    print("4/5: Evaluating model on held-out test set...")
    y_prob_2d = full_pipeline.predict_proba(X_test)
    
    # Extract 1D array using matrix transpose (.T) to ensure 1D shape (2000,)
    y_pred_proba = np.take(y_prob_2d, 1, axis=1)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4)
    }
    
    print("\n--- Test Set Performance ---")
    for k, v in metrics.items():
        print(f" - {k.upper():12s}: {v:.4f}")
        
    # -------------------------------------------------------------
    # 5. MLflow Tracking & Artifact Generation
    # -------------------------------------------------------------
    print("\n5/5: Logging experiment, metrics, and SHAP explainability...")
    mlflow.set_experiment("retail_churn_propensity")
    
    with mlflow.start_run(run_name="xgboost_production_pipeline"):
        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)
        
        # 1. ROC Curve Plot
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', color='navy', lw=2)
        # After
        x_diag = np.linspace(0, 1, 2)
        plt.plot(x_diag, x_diag, linestyle='--', color='gray', label='Random Chance')
        plt.title('ROC Curve - Customer Churn')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        roc_plot_path = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
        plt.savefig(roc_plot_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(roc_plot_path)
        
        # 2. Extract Encoded Feature Names
        cat_encoder = full_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_cols))
        all_feature_names = numeric_cols + cat_feature_names
        
        # 3. Feature Importance Plot
        classifier = full_pipeline.named_steps['classifier']
        importances = classifier.feature_importances_
        fi_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        fi_df = fi_df.sort_values('importance', ascending=False).head(12)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(data=fi_df, x='importance', y='feature', palette='Blues_r')
        plt.title('Top 12 Features Driving Churn')
        fi_plot_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
        plt.savefig(fi_plot_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(fi_plot_path)
        
        # 4. Global SHAP Explanations
        print("Computing SHAP summary values...")
        X_test_transformed = full_pipeline.named_steps['preprocessor'].transform(X_test)
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test_transformed[:500])
        
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test_transformed[:500], feature_names=all_feature_names, show=False)
        shap_plot_path = os.path.join(ARTIFACTS_DIR, "shap_summary.png")
        plt.savefig(shap_plot_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(shap_plot_path)
        
        # 5. Save Unified Artifacts for Streamlit App
        joblib.dump(full_pipeline, os.path.join(ARTIFACTS_DIR, "full_pipeline.joblib"))
        
        with open(os.path.join(ARTIFACTS_DIR, "feature_metadata.json"), "w") as f:
            json.dump({
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "all_feature_names": all_feature_names
            }, f, indent=2)
            
        with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"\nSaved production pipeline to: {os.path.join(ARTIFACTS_DIR, 'full_pipeline.joblib')}")

if __name__ == "__main__":
    train_model()