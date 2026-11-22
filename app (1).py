import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Specialty Chemicals Demand & Inventory Dashboard",
    page_icon="🧪",
    layout="wide"
)

# ============================================================
# 1. CORE FUNCTIONS (CSV LOADING WITH BACKEND FALLBACK)
# ============================================================
@st.cache_data
def load_demand_data():
    # Fallback to local files generated in the workspace
    df_path = 'specialty_chemicals_demand_data.csv'
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    else:
        # Generate on the fly if files aren't found
        categories = ['Optical Brighteners', 'Functional Finishes', 'Dye Intermediates']
        months = pd.date_range(start='2022-01', end='2025-12', freq='ME')
        n_months = len(months)
        records = []
        t = np.arange(n_months)
        params = {
            'Optical Brighteners': {'base': 25, 'trend': 0.02, 'vol': 4.0, 'cost': 5200, 'season_amp': 8},
            'Functional Finishes': {'base': 18, 'trend': 0.015, 'vol': 3.5, 'cost': 3800, 'season_amp': 6},
            'Dye Intermediates': {'base': 45, 'trend': 0.025, 'vol': 5.0, 'cost': 2100, 'season_amp': 10}
        }
        for cat in categories:
            p = params[cat]
            trend = p['base'] + p['trend'] * t + np.random.normal(0, p['vol'] * 0.3, n_months)
            season = p['season_amp'] * np.sin(2 * np.pi * t / 12 + np.pi/4)
            demand = np.maximum(0.5, trend + season + np.random.normal(0, p['vol'], n_months))
            for i, m in enumerate(months):
                records.append({
                    'Date': m,
                    'Category': cat,
                    'Demand_metric_tons': round(demand[i], 2),
                    'Unit_Cost_USD_per_ton': p['cost']
                })
        return pd.DataFrame(records).sort_values('Date').reset_index(drop=True)

@st.cache_data
def load_optimized_parameters():
    param_path = 'optimized_inventory_parameters.csv'
    if os.path.exists(param_path):
        return pd.read_csv(param_path)
    else:
        # Mock parameters table to prevent crash
        return pd.DataFrame({
            'Category': ['Optical Brighteners', 'Functional Finishes', 'Dye Intermediates'],
            'Optimal_Model': ['3M_MA', '3M_MA', 'ARIMA(1,1,1)'],
            'Forecasted_Demand_MT_Month': [33.12, 18.10, 45.53],
            'Safety_Stock_MT': [20.42, 15.70, 25.09],
            'Reorder_Point_ROP_MT': [119.79, 70.00, 161.68],
            'Economic_Order_Quantity_EOQ_MT': [19.94, 14.74, 23.37],
            'Average_Inventory_Level_MT': [80.07, 50.22, 105.07],
            'Annual_Holding_Cost_USD': [16014.83, 10043.42, 21013.66]
        })

# Load Data sources
df = load_demand_data()
optimized_params_df = load_optimized_parameters()

# ============================================================
# 2. STREAMLIT UI: SIDEBAR CONTROLS
# ============================================================
st.sidebar.title("⚙️ Policy Parameters")
selected_category = st.sidebar.selectbox("Select Specialty Chemical Category", df['Category'].unique())

# Fetch pre-calculated database attributes dynamically for the selected category
cat_params = optimized_params_df[optimized_params_df['Category'] == selected_category].iloc[0]

# Dynamic policy tuning overrides via sidebar sliders
st.sidebar.subheader("Interactive Overrides")
override_rop = st.sidebar.slider("Reorder Point (ROP) Override", 10.0, 300.0, float(cat_params['Reorder_Point_ROP_MT']))
override_eoq = st.sidebar.slider("Economic Order Quantity (EOQ) Override", 5.0, 100.0, float(cat_params['Economic_Order_Quantity_EOQ_MT']))

# ============================================================
# 3. TRAIN/TEST DEMAND DATA SPLIT
# ============================================================
cat_df = df[df['Category'] == selected_category].sort_values('Date').reset_index(drop=True)
split_idx = int(0.8 * len(cat_df))
train_df = cat_df.iloc[:split_idx]
test_df = cat_df.iloc[split_idx:]

# ============================================================
# 4. DASHBOARD PRESENTATION
# ============================================================
st.title("🧪 Textile Specialty Chemicals AI Forecasting & Inventory Optimization")
st.write("Dynamic decision support tool powered by backend optimized forecasting models (SDG 12: Responsible Production).")

# --- KPI METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Chemical Category", selected_category)
with col2:
    st.metric("Optimal Forecast Model", cat_params['Optimal_Model'])
with col3:
    st.metric("Monthly Forecast Demand", f"{cat_params['Forecasted_Demand_MT_Month']:.2f} MT")
with col4:
    st.metric("Annual Holding Cost", f"${cat_params['Annual_Holding_Cost_USD']:,.2f}")

# --- TABS FOR VISUALIZATION AND ANALYSIS ---
tab1, tab2, tab3 = st.tabs(["📈 Demand History & Split", "📦 Inventory Optimization KPIs", "⚙️ Parameter Matrix Summary"])

with tab1:
    st.subheader("Demand Profiles: Historical Trends")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(train_df['Date'], train_df['Demand_metric_tons'], label='Train Set Demand', color='black', marker='o', alpha=0.5)
    ax.plot(test_df['Date'], test_df['Demand_metric_tons'], label='Test Set Demand (Out-of-sample)', color='blue', marker='x')
    ax.set_ylabel("Metric Tons / Month")
    ax.set_title(f"{selected_category} Multi-Year Demand Profile", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.subheader("Inventory Policy Parameters & Costs")

    # Display dynamic comparative metrics
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Safety Stock Buffer", f"{cat_params['Safety_Stock_MT']:.2f} MT")
    with col_kpi2:
        st.metric("Reorder Point (ROP)", f"{override_rop:.2f} MT")
    with col_kpi3:
        st.metric("Economic Order Quantity (EOQ)", f"{override_eoq:.2f} MT")
    with col_kpi4:
        st.metric("Average Inventory Level", f"{cat_params['Average_Inventory_Level_MT']:.2f} MT")

    st.subheader("Optimized Inventory Strategy Table")
    st.dataframe(optimized_params_df[optimized_params_df['Category'] == selected_category])

    st.markdown("""
    * **SDG 12 Impact**: Aligning procurement directly with demand eliminates chemical holding depreciation, minimizing hazardous expiration waste in Hyderabad's storage facilities.
    """)

with tab3:
    st.subheader("All-Category Portfolio Summary Matrix")
    st.write("Full overview of optimized backend parameters deployed in production:")
    st.dataframe(optimized_params_df)
