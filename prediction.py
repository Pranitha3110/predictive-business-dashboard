import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(df, feature_cols, target_col):
    clean_df = df[feature_cols + [target_col]].dropna()
    X = clean_df[feature_cols]
    y = clean_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}).reset_index(drop=True)
    return model, mae, r2, eval_df