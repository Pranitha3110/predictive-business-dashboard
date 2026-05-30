import streamlit as st
import pandas as pd
import plotly.express as px

# Import your utility modules
from utils.preprocessing import clean_data
from utils.visualization import create_bar_chart, create_line_chart, create_scatter_plot
from utils.prediction import train_model
from utils.insights import generate_basic_insights

# 1. Global Page Layout Setup
st.set_page_config(page_title="AI Business Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("AI-Powered Business Analytics Dashboard")
st.markdown("Upload your raw business data below to clean, visualize, and extract predictive insights instantly.")

# 2. Sidebar File Upload System
with st.sidebar:
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload your business dataset (CSV format)", type=["csv"])
    
    st.markdown("---")
    st.markdown("### Project Progress")
    st.info("Phase 1-6 fully operational. Transition tabs below to explore data properties.")

# 3. Main Dashboard Engine Execution
if uploaded_file:
    # Read data safely
    raw_df = pd.read_csv(uploaded_file)
    
    # Silently clean the dataset via your preprocessing script
    df = clean_data(raw_df)
    
    # Separate application sections visually via Native Streamlit Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        " Data Preview & EDA", 
        " Interactive Visualizations", 
        " Predictive Modeling", 
        " Automated AI Insights"
    ])
    
    # --- TAB 1: DATA PREVIEW & EXPLORATORY ANALYSIS ---
    with tab1:
        st.header("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Records (Rows)", value=df.shape[0])
        with col2:
            st.metric(label="Total Features (Columns)", value=df.shape[1])
            
        st.subheader("Cleaned Dataset Snapshot")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Statistical Summary Profile")
        st.dataframe(df.describe().T, use_container_width=True)
        
    # --- TAB 2: INTERACTIVE CHARTS ---
    # --- TAB 2: INTERACTIVE CHARTS ---
    with tab2:
        st.header("Business Performance Visualizations")
        
        # Segment columns by their data type properties
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Best choices for X-axis are categories or timeline fields
        x_axis_options = categorical_columns + date_columns if (categorical_columns + date_columns) else df.columns.tolist()

        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            chart_type = st.selectbox("1. Choose Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot"])
        with v_col2:
            # Dropdowns will naturally default to cleaner dimensional features
            x_axis = st.selectbox("2. Select X-Axis Dimensional Data", x_axis_options)
        with v_col3:
            y_axis = st.selectbox("3. Select Y-Axis Metric Data (Numeric)", numeric_columns)
            
        st.markdown("---")
        
        # Guard clause: check if user picked a text column with too many unique values (like full reviews)
        if df[x_axis].nunique() > 30 and chart_type in ["Bar Chart", "Line Chart"]:
            st.warning(f"⚠️ **'{x_axis}'** contains too many unique text rows ({df[x_axis].nunique()}) to display cleanly. Try grouping the data or choosing a column with fewer categories (like ratings, genres, or years).")
        else:
            if chart_type == "Bar Chart":
                fig = create_bar_chart(df, x_axis, y_axis)
            elif chart_type == "Line Chart":
                fig = create_line_chart(df, x_axis, y_axis)
            elif chart_type == "Scatter Plot":
                fig = create_scatter_plot(df, x_axis, y_axis)
                
            st.plotly_chart(fig, use_container_width=True)
        
    # --- TAB 3: MACHINE LEARNING PREDICTIONS ---
    with tab3:
        st.header("Predictive Modeling Engine")
        st.write("Train a supervised linear regression model to predict continuous target variables.")
        
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        
        ml_col1, ml_col2 = st.columns(2)
        with ml_col1:
            feature_cols = st.multiselect("Select Feature Variables (X)", numeric_columns)
        with ml_col2:
            target_col = st.selectbox("Select Target Variable to Predict (Y)", numeric_columns)
            
        if st.button("Run Predictive Model"):
            if not feature_cols:
                st.error("Please pick at least one feature variable to begin training.")
            elif target_col in feature_cols:
                st.error("The target variable cannot simultaneously be an input feature.")
            else:
                with st.spinner("Splitting data and optimizing model coefficients..."):
                    model, mae, r2, eval_df = train_model(df, feature_cols, target_col)
                    
                    st.success("Model Trained Successfully!")
                    
                    # Display performance KPIs
                    m_metric1, m_metric2 = st.columns(2)
                    with m_metric1:
                        st.metric(label="Mean Absolute Error (MAE)", value=f"{round(mae, 4)}")
                    with m_metric2:
                        st.metric(label="R-Squared Score ($R^2$)", value=f"{round(r2, 4)}")
                        
                    # Visualize predictions vs reality
                    st.subheader("Model Evaluation: Actual vs. Predicted Values")
                    fig_eval = px.scatter(eval_df, x='Actual', y='Predicted', 
                                         trendline="ols", trendline_color_override="red",
                                         template="plotly_white")
                    st.plotly_chart(fig_eval, use_container_width=True)

    # --- TAB 4: AUTOMATED AI INSIGHTS ---
    with tab4:
        st.header("Automated Business Narrative Insights")
        st.write("Below are natural language analytical takeaways computed from your structural data distributions:")
        
        insights = generate_basic_insights(df)
        
        for insight in insights:
            st.markdown(f"- {insight}")

else:
    # Warm welcome state when no file is present
    st.info("Welcome! Please upload a valid CSV dataset file in the left sidebar menu to initialize analysis.")