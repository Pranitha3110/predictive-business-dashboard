import pandas as pd

def generate_basic_insights(df):
    insights = []
    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in numeric_cols:
        insights.append(f"The average **{col}** across the dataset is **{round(df[col].mean(), 2)}**.")
        insights.append(f"The maximum recorded value for **{col}** is **{round(df[col].max(), 2)}**.")
        
    for col in categorical_cols:
        top_value = df[col].mode()
        if not top_value.empty:
            count_top = df[col].value_counts().iloc[0]
            percentage = (count_top / len(df)) * 100
            insights.append(f"Category **'{top_value[0]}'** is the most frequent in **{col}** ({round(percentage, 1)}% of records).")
            
    return insights