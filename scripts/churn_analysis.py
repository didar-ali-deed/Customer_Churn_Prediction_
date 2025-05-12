import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("First 5 rows of dataset:")
print(data.head())

# Check shape (rows, columns)
print("\nDataset Shape:", data.shape)

# List columns
print("\nColumns:", data.columns.tolist())

# Check data types
print("\nData Types:\n", data.dtypes)

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Summarize numerical features
print("\nNumerical Features Summary:\n", data.describe())

# Visualize churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.savefig('../results/churn_distribution.png')
plt.close()

# Visualize churn by contract type
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=data)
plt.title('Churn by Contract Type')