import pandas as pd

def clean_data(df):
    """
    Cleans the input dataframe by removing duplicates, handling missing values,
    and attempting basic data type conversions.
    """
    # 1. Drop complete duplicates
    df = df.drop_duplicates()
    
    # 2. Fill empty values based on logic criteria
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
            
    # 3. Handle dates automatically
    for col in df.columns:
        if df[col].dtype == 'object':
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    pass
                    
    return df