import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set page title and layout
st.set_page_config(page_title="HRlytics Dashboard", layout="wide")

# Title and description
st.title("HRlytics Dashboard")
st.markdown("This dashboard provides insights into employee attrition and key HR metrics.")

# Load Data
@st.cache_data
def load_data():
    # Use relative path assuming execution from dashboard/ or root
    # Try different paths
    for path in ['../data/hr_data.csv', 'data/hr_data.csv']:
        if os.path.exists(path):
            return pd.read_csv(path)
    st.error("Data file not found! Please check 'data/hr_data.csv'")
    return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Filter Options")
    
    # Example filters
    if 'Department' in df.columns:
        dept = st.sidebar.multiselect("Select Department", df['Department'].unique(), default=df['Department'].unique())
        df_filtered = df[df['Department'].isin(dept)]
    else:
        df_filtered = df.copy()

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", len(df_filtered))
    
    if 'Attrition' in df.columns:
        attrition_rate = (df_filtered['Attrition'] == 'Yes').mean() * 100
        col2.metric("Attrition Rate", f"{attrition_rate:.1f}%")
        
    if 'Age' in df.columns:
        avg_age = df_filtered['Age'].mean()
        col3.metric("Average Age", f"{avg_age:.1f}")

    if 'MonthlyIncome' in df.columns:
        avg_income = df_filtered['MonthlyIncome'].mean()
        col4.metric("Avg Monthly Income", f"${avg_income:,.0f}")

    # Data Overview
    st.subheader("Employee Data Overview")
    st.dataframe(df_filtered.head(10))

    # Visualizations
    st.subheader("Visual Analysis")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if 'Attrition' in df.columns and 'Department' in df.columns:
            st.markdown("#### Attrition by Department")
            try:
                fig, ax = plt.subplots()
                sns.countplot(data=df_filtered, x='Department', hue='Attrition', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except Exception as e:
                st.write("Could not generate chart.")
                
    with col_chart2:
         if 'JobSatisfaction' in df.columns:
            st.markdown("#### Job Satisfaction Distribution")
            try:
                fig2, ax2 = plt.subplots()
                sns.histplot(df_filtered['JobSatisfaction'], kde=True, ax=ax2)
                st.pyplot(fig2)
            except Exception as e:
                st.write("Could not generate chart.")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    
    # Select numeric columns only
    numeric_df = df_filtered.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        try:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
            st.pyplot(fig3)
        except Exception as e:
            st.write("Could not generate heatmap.")
    else:
        st.write("Not enough numeric data for correlation calculation.")

else:
    st.write("Data failed to load.")
