import pandas as pd
import numpy as np

# Load sample dataset
df = pd.read_csv('health_data.csv')

# Handle missing values
df['age'].fillna(df['age'].mean(), inplace=True)

# Feature encoding (convert categorical to numerical)
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# Add a derived feature
df['bmi_category'] = np.where(df['bmi'] >= 30, 'obese', 'non-obese')
# nnw
# Normalize numerical columns
cols_to_normalize = ['age', 'bmi']
df[cols_to_normalize] = (df[cols_to_normalize] - df[cols_to_normalize].mean()) / df[cols_to_normalize].std()
# Data cleaning, feature engineering, and exploratory data analysis (EDA).
print(df.head())
