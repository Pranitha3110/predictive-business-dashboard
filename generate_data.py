import pandas as pd
import numpy as np

# Set random seed for consistent business trends
np.random.seed(42)
n_rows = 500

# Generate a mock retail sales dataset
data = {
    'Date': pd.date_range(start='2025-01-01', periods=n_rows, freq='D'),
    'Region': np.random.choice(['North America', 'Europe', 'Asia-Pacific', 'Latin America'], n_rows),
    'Category': np.random.choice(['Electronics', 'Clothing', 'Home Appliances', 'Office Supplies'], n_rows),
    'Marketing_Spend': np.random.uniform(100, 1500, n_rows).round(2),
    'Discount_Percent': np.random.choice([0, 5, 10, 15, 20], n_rows),
}

df = pd.DataFrame(data)

# Add mathematical relationships so the Machine Learning model works perfectly
df['Units_Sold'] = (10 + (df['Marketing_Spend'] * 0.05) - (df['Discount_Percent'] * 0.2) + np.random.normal(0, 5, n_rows)).astype(int)
df['Units_Sold'] = df['Units_Sold'].clip(lower=1) # Ensure no negative sales
df['Revenue'] = (df['Units_Sold'] * np.random.uniform(20, 120, n_rows)).round(2)
df['Profit'] = (df['Revenue'] * np.random.uniform(0.15, 0.40, n_rows)).round(2)

# Save it to your workspace
df.to_csv('Sample_Business_Sales.csv', index=False)
print("🎯 Success! 'Sample_Business_Sales.csv' has been generated in your folder.")