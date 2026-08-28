import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import random

# --------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------
st.set_page_config(page_title="Retail Churn Predictor", layout="wide", page_icon="🛍️")

# --------------------------------------------------------------------
# LOAD MODEL & DATA (Cached)
# --------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("artifacts/full_pipeline.joblib")

@st.cache_resource
def load_reference_data():
    return pd.read_parquet("artifacts/reference_data.parquet")

full_pipeline = load_model()
classifier = full_pipeline.named_steps['classifier']
preprocessor = full_pipeline.named_steps['preprocessor']
reference_df = load_reference_data()

# Feature definitions
categorical_cols = ['gender', 'state', 'acquisition_channel', 'membership_tier']
numeric_cols = ['age', 'is_email_opted_in', 'tenure_days', 'frequency', 'monetary_value', 
                'avg_order_value', 'total_units_purchased', 'avg_discount_pct', 
                'discount_order_share', 'total_returns', 'return_rate', 'avg_shipping_days', 
                'max_shipping_days', 'web_order_share', 'app_order_share', 'store_order_share', 
                'distinct_categories_bought', 'apparel_item_count', 'electronics_item_count']

cat_encoder = preprocessor.named_transformers_['cat']
encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
all_feature_names = numeric_cols + encoded_cat_names

# SHAP Explainer
background_sample = reference_df.sample(n=100, random_state=42)
background_transformed = preprocessor.transform(background_sample).astype(np.float64)

@st.cache_resource
def load_explainer():
    return shap.Explainer(classifier, background_transformed, feature_names=all_feature_names)

explainer = load_explainer()

# --------------------------------------------------------------------
# FIX 1: PRE-SAMPLE 5 REAL CUSTOMERS (for auto-fill honesty)
# --------------------------------------------------------------------
@st.cache_resource
def get_base_customers():
    """Return 5 real customer rows to use as auto-fill baselines."""
    return reference_df.sample(n=5, random_state=42).to_dict('records')

base_customers = get_base_customers()

# Initialize session state for the current base customer
if "current_customer_idx" not in st.session_state:
    st.session_state.current_customer_idx = 0

def shuffle_customer():
    """Cycle to the next pre-sampled customer."""
    st.session_state.current_customer_idx = (st.session_state.current_customer_idx + 1) % len(base_customers)

# --------------------------------------------------------------------
# CREATE TABS
# --------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📖 Full Case Study", "⚙️ Tech Stack", "🎯 Live Demo"])

# ================================================================
# TAB 1: FULL CASE STUDY
# ================================================================
with tab1:
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:12px 20px; border-radius:8px; margin-bottom:24px; border:1px solid #d0d5dc; color:#212529;">
        <b style="font-size:1.1rem;">📑 On this page:</b> &nbsp;&nbsp;
        <a href="#hero">Overview</a> &bull;
        <a href="#business">Problem</a> &bull;
        <a href="#data">Data</a> &bull;
        <a href="#architecture">Architecture</a> &bull;
        <a href="#modeling">Modeling</a> &bull;
        <a href="#validation">Validation</a> &bull;
        <a href="#closing">Skills</a>
    </div>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown('<a id="hero"></a>', unsafe_allow_html=True)
    st.title("🛍️ Retail Customer Churn Prediction")
    st.markdown("""
    ### *Predicting which customers will churn before they leave, using behavioral and transactional signals*
    
    **The goal:** Identify at-risk customers early so the retention team can intervene with targeted offers.
    
    <div style="margin-top: 12px;">
        <span style="background-color:#e9ecef; padding:4px 12px; border-radius:20px; font-size:0.85rem; display:inline-block; margin-right:8px;color:#212529">🔮 XGBoost</span>
        <span style="background-color:#e9ecef; padding:4px 12px; border-radius:20px; font-size:0.85rem; display:inline-block; margin-right:8px;color:#212529">📊 SHAP</span>
        <span style="background-color:#e9ecef; padding:4px 12px; border-radius:20px; font-size:0.85rem; display:inline-block; margin-right:8px;color:#212529">☁️ BigQuery</span>
        <span style="background-color:#e9ecef; padding:4px 12px; border-radius:20px; font-size:0.85rem; display:inline-block; margin-right:8px;color:#212529">🌊 Streamlit</span>
        <span style="background-color:#e9ecef; padding:4px 12px; border-radius:20px; font-size:0.85rem; display:inline-block; margin-right:8px;color:#212529">🧪 MLflow</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Business Problem
    st.markdown('<a id="business"></a>', unsafe_allow_html=True)
    st.subheader("💼 1. The Business Problem")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Why churn matters:**  
        Acquiring a new customer costs **5x more** than retaining an existing one. For a mid-sized retailer, losing just 5% of customers can reduce profitability by **25–30%**.
        
        **The challenge:**  
        Only **~20%** of customers actually churn. This severe class imbalance means a naive model that predicts "everyone stays" would achieve 80% accuracy—but would be completely useless.
        """)
    with col2:
        st.metric("📉 Churn Rate", "20%", delta="Minority class", delta_color="off")
        st.metric("💸 Cost of Acquisition", "5x", delta="vs. Retention")
    st.divider()

    # Data
    st.markdown('<a id="data"></a>', unsafe_allow_html=True)
    st.subheader("🗃️ 2. The Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Source:**  
        Google BigQuery `retail_analytics.fct_customer_churn_features`
        
        **Size:**  
        - 10,000 customer records  
        - 23 features (19 numeric, 4 categorical)
        
        **Target:** `is_churned` (1 = left, 0 = stayed)
        """)
    with col2:
        st.markdown("""
        **📌 Honest note on synthetic data:**  
        This dataset is synthetically generated to simulate real-world retail transactions. I built it this way to demonstrate my end-to-end pipeline skills without proprietary data.
        
        The ML pipeline, modeling choices, and deployment architecture are identical to what I would use with real production data.
        """)
    st.dataframe(reference_df.head(5), use_container_width=True)
    st.divider()

    # Architecture
    st.markdown('<a id="architecture"></a>', unsafe_allow_html=True)
    st.subheader("🏗️ 3. End-to-End Architecture")
    steps = [
        ("🗂️ Data Generation", "Synthetic retail data"),
        ("☁️ BigQuery", "Data warehouse"),
        ("🛠️ Feature Engineering", "One-hot encoding + pipelines"),
        ("🤖 XGBoost", "Model training + tuning"),
        ("📊 SHAP", "Explainability"),
        ("🌊 Streamlit", "Interactive dashboard")
    ]
    cols = st.columns(len(steps))
    for i, (col, (emoji, label)) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:16px 8px; border-radius:8px; text-align:center; border:1px solid #d0d5dc; height:100px; display:flex; flex-direction:column; justify-content:center; color:#212529;">
                <div style="font-size:2rem;">{emoji}</div>
                <div style="font-weight:600; font-size:0.9rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            if i < len(steps) - 1:
                st.markdown("<p style='text-align:center; font-size:1.5rem; margin:0;'>➡️</p>", unsafe_allow_html=True)
    st.divider()

    # Modeling
    st.markdown('<a id="modeling"></a>', unsafe_allow_html=True)
    st.subheader("🧠 4. Modeling Approach")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Algorithm: XGBoost**  
        - Industry standard for tabular data.
        - Handles non-linear relationships and feature interactions.
        - Native SHAP integration for explainability.
        
        **Class imbalance handling:**  
        Used `scale_pos_weight = (non_churners / churners)` to force the model to pay attention to churners.
        """)
    with col2:
        st.markdown("""
        **🔧 Key hyperparameters:**
        - `n_estimators`: 150
        - `learning_rate`: 0.05
        - `max_depth`: 4
        - `scale_pos_weight`: 4.0
        """)
    st.divider()

    # Validation
    st.markdown('<a id="validation"></a>', unsafe_allow_html=True)
    st.subheader("📈 5. Model Validation")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.metric("🏆 ROC-AUC", "0.9229")
    with m2: st.metric("📊 PR-AUC", "0.8622")
    with m3: st.metric("✅ Accuracy", "0.8175")
    with m4: st.metric("🎯 Precision", "0.6683")
    with m5: st.metric("🔍 Recall", "0.8549")
    with m6: st.metric("⚖️ F1-Score", "0.7502")
    
    # FIX 3: Honest ROI math
    st.markdown("""
    **Business impact (derived from this dataset):**  
    - Dataset: 10,000 customers, churn rate ~20% → ~2,000 churners total.  
    - Model recall: 85.5% → catches ~1,710 of those churners.  
    - If we retain 30% of those → ~513 customers saved.  
    - At $500/year average customer value → **~$256,500/year in retained revenue**.
    """)
    st.divider()

    # Closing
    st.markdown('<a id="closing"></a>', unsafe_allow_html=True)
    st.subheader("🎯 6. What This Demonstrates")
    st.markdown("""
    | Skill Area | What I built | Why it matters |
    | :--- | :--- | :--- |
    | **Data Engineering** | BigQuery mart, pandas preprocessing | I can acquire, clean, and store data. |
    | **ML Modeling** | XGBoost with class imbalance handling | I build production-grade models. |
    | **MLOps** | MLflow for experiment tracking | My work is reproducible and auditable. |
    | **Explainability** | SHAP for global/local explanations | I can translate black-box models into business trust. |
    | **Deployment** | Streamlit dashboard with real-time predictions | I can ship working software for non-technical users. |
    | **Communication** | This full case study | I bridge data science and the C-suite. |
    """)

# ================================================================
# TAB 2: TECH STACK
# ================================================================
with tab2:
    st.title("⚙️ Tech Stack & Tools")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("☁️ Data Layer")
        st.markdown("- Google BigQuery\n- pandas\n- Parquet")
    with col2:
        st.subheader("🧠 Modeling Layer")
        st.markdown("- Scikit-learn\n- XGBoost\n- SHAP")
    with col3:
        st.subheader("🚀 MLOps & Deployment")
        st.markdown("- MLflow\n- Streamlit\n- Joblib")
    st.divider()
    st.subheader("🧠 Key Decisions")
    decisions = [
        ("No Feature Scaling", "XGBoost is tree-based; it doesn't require scaling."),
        ("One-Hot Encoding", "Avoids XGBoost's `enable_categorical` flag, which SHAP doesn't fully support."),
        ("SHAP for Explainability", "Critical for business trust."),
        ("MLflow for Tracking", "Ensures reproducibility."),
        ("Streamlit for Deployment", "Rapid interactive demos without frontend code.")
    ]
    for decision, explanation in decisions:
        st.markdown(f"**{decision}** — *{explanation}*")

# ================================================================
# TAB 3: LIVE DEMO
# ================================================================
with tab3:
    st.title("🎯 Live Churn Predictor")
    st.markdown("#### *Adjust the 4 key inputs below to see how customer attributes impact churn risk.*")
    st.caption("💡 The remaining features are auto-filled from a randomly selected real customer profile.")

    # ------------------------------------------------------------------
    # FIX 1: Shuffle button + get current base customer
    # ------------------------------------------------------------------
    col_shuffle, _ = st.columns([1, 5])
    with col_shuffle:
        if st.button("🎲 Shuffle background customer"):
            shuffle_customer()
    
    base_customer = base_customers[st.session_state.current_customer_idx]
    
    # ------------------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------------------
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        tenure = st.number_input("📅 Tenure (Days)", min_value=0, max_value=2000, value=180, step=30)
        monetary_value = st.number_input("💰 Total Monetary Value ($)", min_value=0, max_value=50000, value=1500, step=500)
        frequency = st.slider("🛒 Purchase Frequency (Orders)", min_value=1, max_value=50, value=10, step=1)
    
    with col_input2:
        return_rate = st.slider("📦 Return Rate (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0) / 100.0
        membership = st.selectbox("🏅 Membership Tier", options=['Bronze', 'Silver', 'Gold', 'Platinum'])
        
        # FIX 1 & 4: Show ALL hidden fields from the actual base customer
        with st.expander("⚙️ Auto-filled fields (from a real customer record)"):
            # Show all fields that are NOT the 4 primary inputs
            hidden_fields = {
                k: v for k, v in base_customer.items() 
                if k not in ['tenure_days', 'monetary_value', 'frequency', 'return_rate', 'membership_tier']
            }
            # Drop 'is_churned' from the display so it doesn't confuse viewers
            clean_hidden = {k: v for k, v in hidden_fields.items() if k != 'is_churned'}
            st.json(clean_hidden)
            st.caption("📌 These are the actual values from the selected real customer record. is_churned (the donor's real outcome) is hidden to avoid confusion with the current prediction.")

            st.caption(f"🎲 Currently using Customer #{st.session_state.current_customer_idx + 1} of {len(base_customers)}. Click 'Shuffle background customer' to swap.")
    
    # ------------------------------------------------------------------
    # BUILD INPUT DATAFRAME (using base_customer for hidden fields)
    # ------------------------------------------------------------------
    user_input = {
        'age': base_customer['age'],
        'is_email_opted_in': base_customer['is_email_opted_in'],
        'tenure_days': tenure,
        'frequency': frequency,
        'monetary_value': monetary_value,
        'avg_order_value': base_customer['avg_order_value'],
        'total_units_purchased': base_customer['total_units_purchased'],
        'avg_discount_pct': base_customer['avg_discount_pct'],
        'discount_order_share': base_customer['discount_order_share'],
        'total_returns': base_customer['total_returns'],
        'return_rate': return_rate,
        'avg_shipping_days': base_customer['avg_shipping_days'],
        'max_shipping_days': base_customer['max_shipping_days'],
        'web_order_share': base_customer['web_order_share'],
        'app_order_share': base_customer['app_order_share'],
        'store_order_share': base_customer['store_order_share'],
        'distinct_categories_bought': base_customer['distinct_categories_bought'],
        'apparel_item_count': base_customer['apparel_item_count'],
        'electronics_item_count': base_customer['electronics_item_count'],
        'gender': base_customer['gender'],
        'state': base_customer['state'],
        'acquisition_channel': base_customer['acquisition_channel'],
        'membership_tier': membership
    }
    
    input_df = pd.DataFrame([user_input])
    
    # ------------------------------------------------------------------
    # PREDICTION & EXPLANATION
    # ------------------------------------------------------------------
    pred_proba = full_pipeline.predict_proba(input_df)[0][1]
    input_transformed = preprocessor.transform(input_df).astype(np.float64)
    explanation = explainer(input_transformed)
    exp = explanation[0]
    
    # ------------------------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------------------------
    st.divider()
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        st.subheader("🧠 Churn Risk Assessment")
        if pred_proba >= 0.7:
            st.error(f"### ⚠️ HIGH RISK: {pred_proba:.1%}")
            st.write("🚨 Immediate retention offer recommended.")
        elif pred_proba >= 0.4:
            st.warning(f"### 🔶 MODERATE RISK: {pred_proba:.1%}")
            st.write("👀 Monitor closely.")
        else:
            st.success(f"### ✅ LOW RISK: {pred_proba:.1%}")
            st.write("😊 This customer appears loyal and engaged.")
    
    with col_result2:
        st.subheader("📊 Top Drivers for this Customer")
        
        # FIX 2: SHAP log-odds caption
        st.caption("🔵 **Blue bars** lower churn risk • 🔴 **Red bars** raise it • Longer = bigger impact")
        st.caption("📐 Values shown are in **log-odds units** — direction (red/blue) matters more than the exact number.")
        
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='none')
        ax.set_facecolor('none')
        shap.waterfall_plot(exp, max_display=8, show=False)
        # Add a visible border so it looks like a card
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')  # Dark grey border
            spine.set_linewidth(1)
        shap.waterfall_plot(exp, max_display=8, show=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('#d0d5dc')
            spine.set_linewidth(1)
        plt.tight_layout()
        st.pyplot(fig, bbox_inches='tight')
        plt.close()
    
    st.divider()
    st.caption("🚀 Built with XGBoost, SHAP, MLflow, and Streamlit • Achieves 0.92 ROC-AUC • Full code on GitHub.")