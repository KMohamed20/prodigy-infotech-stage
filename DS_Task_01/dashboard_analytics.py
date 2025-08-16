"""
Data Science Task 01 - Business Analytics Dashboard
Prodigy InfoTech Internship | Khalid Ag Mohamed Aly
August 2025

Objective: Create an interactive business analytics dashboard with Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Business Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Business Performance Dashboard")
st.markdown("### Real-time KPIs and Analytics | Prodigy InfoTech Internship")

# Generate synthetic data
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
regions = ['Nord', 'Sud', 'Est', 'Ouest', 'Centre']
products = ['ProductA', 'ProductB', 'ProductC']

data = []
for date in dates:
    for region in regions:
        for product in products:
            sales = np.random.normal(1000, 300) * (1 + np.sin(date.month * np.pi / 6) * 0.5)
            sales = max(sales, 200)
            customers = int(sales / 50 + np.random.normal(10, 3))
            cost = sales * np.random.uniform(0.4, 0.7)
            profit = sales - cost
            roi = (profit / (cost + 1)) * 100
            
            data.append({
                'Date': date,
                'Region': region,
                'Product': product,
                'Sales': round(sales, 2),
                'Customers': max(customers, 1),
                'Cost': round(cost, 2),
                'Profit': round(profit, 2),
                'ROI': round(roi, 2)
            })

df = pd.DataFrame(data)

# Sidebar filters
st.sidebar.header("🔍 Filters")
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=regions,
    default=regions
)

selected_products = st.sidebar.multiselect(
    "Select Products",
    options=products,
    default=products
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=[dates[0], dates[-1]]
)

# Apply filters
filtered_df = df[
    (df['Region'].isin(selected_regions)) &
    (df['Product'].isin(selected_products)) &
    (df['Date'] >= pd.Timestamp(date_range[0])) &
    (df['Date'] <= pd.Timestamp(date_range[1]))
]

# KPIs
col1, col2, col3, col4 = st.columns(4)

total_sales = filtered_df['Sales'].sum()
total_customers = filtered_df['Customers'].sum()
avg_profit = filtered_df['Profit'].mean()
avg_roi = filtered_df['ROI'].mean()

col1.metric("💰 Total Sales", f"€{total_sales:,.0f}")
col2.metric("👥 Total Customers", f"{total_customers:,}")
col3.metric("💼 Avg Profit", f"€{avg_profit:.0f}")
col4.metric("📊 Avg ROI", f"{avg_roi:.1f}%")

# Charts
st.markdown("### 📈 Performance Analysis")

fig1 = px.line(
    filtered_df.groupby('Date')['Sales'].sum().reset_index(),
    x='Date',
    y='Sales',
    title='Daily Sales Evolution',
    labels={'Sales': 'Revenue (€)'},
    height=400
)
st.plotly_chart(fig1, use_container_width=True)

col1, col2 = st.columns(2)

# Regional performance
regional_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
fig2 = px.bar(
    regional_sales,
    x='Region',
    y='Sales',
    title='Sales by Region',
    color='Region',
    text='Sales',
    height=400
)
fig2.update_traces(texttemplate='€%{text:,.0f}', textposition='outside')
col1.plotly_chart(fig2, use_container_width=True)

# Product distribution
product_sales = filtered_df.groupby('Product')['Sales'].sum().reset_index()
fig3 = px.pie(
    product_sales,
    names='Product',
    values='Sales',
    title='Sales Distribution by Product',
    height=400
)
col2.plotly_chart(fig3, use_container_width=True)

# Additional insights
st.markdown("### 🧠 Business Insights")

insights = [
    f"📈 **Strong Performance**: Total sales reached €{total_sales:,.0f} during selected period.",
    f"🎯 **Regional Leader**: {(regional_sales.loc[regional_sales['Sales'].idxmax(), 'Region'])} outperformed with highest sales.",
    f"💡 **Top Product**: {(product_sales.loc[product_sales['Sales'].idxmax(), 'Product'])} generated the most revenue.",
    f"💰 **Healthy Profitability**: Average ROI of {avg_roi:.1f}% indicates good investment returns.",
    f"👥 **Customer Growth**: Reached {total_customers:,} customers across all regions."
]

for insight in insights:
    st.markdown(f"- {insight}")

# Data table
st.markdown("### 📋 Detailed Data")
st.dataframe(
    filtered_df.sort_values('Date', ascending=False),
    use_container_width=True,
    height=300
)

# Footer
st.markdown("---")
st.markdown("© 2025 Prodigy InfoTech | Machine Learning Internship Project")
