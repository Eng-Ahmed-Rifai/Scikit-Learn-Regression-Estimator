"""
Scikit-Learn Regression & Predictive Estimator: End-to-End Pipeline
Author: Eng. Ahmed Rifai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def generate_business_data(n_samples=4000, random_state=42):
    np.random.seed(random_state)
    marketing = np.random.uniform(2000, 100000, n_samples)
    team = np.random.randint(1, 30, n_samples)
    deal = np.random.uniform(500, 15000, n_samples)
    leads = np.random.uniform(50, 1200, n_samples)
    retention = np.random.uniform(50, 99, n_samples)
    
    # Ground truth formula with non-linear returns & noise
    noise = np.random.normal(0, 3000, n_samples)
    mrr = (
        (marketing ** 0.78 * 12.5)
        + (leads * 0.12 * deal)
        + (team * deal * 3.2)
    ) * (retention / 100) + noise
    
    df = pd.DataFrame({
        'MarketingSpend': marketing,
        'SalesTeam': team,
        'DealValue': deal,
        'InboundLeads': leads,
        'RetentionRate': retention,
        'MRR': mrr
    })
    return df

def train_regression_model():
    df = generate_business_data()
    X = df.drop(columns=['MRR'])
    y = df['MRR']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    
    importances = model.feature_importances_
    for col, imp in zip(X.columns, importances):
        print(f"  Feature [{col}]: {imp:.4f}")

if __name__ == "__main__":
    train_regression_model()
