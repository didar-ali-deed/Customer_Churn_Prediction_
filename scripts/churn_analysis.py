import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def explore_data():
    """Perform exploratory data analysis on the dataset."""
    # Load dataset
    data = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("First 5 rows of dataset:")
    print(data.head())

    # Check shape, columns, types, and missing values
    print("\nDataset Shape:", data.shape)
    print("\nColumns:", data.columns.tolist())
    print("\nData Types:\n", data.dtypes)
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
    plt.savefig('../results/churn_by_contract.png')
    plt.close()

    return data

def preprocess_data(data):
    """Preprocess the dataset for machine learning and save splits."""
    # Handle missing values in TotalCharges
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    # Drop irrelevant column
    data.drop('customerID', axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        data[col] = le.fit_transform(data[col])

    categorical_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Split features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nPreprocessing Complete:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Save preprocessed data
    X_train.to_csv('../data/X_train.csv', index=False)
    X_test.to_csv('../data/X_test.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)
    print("\nSaved preprocessed data to data/ folder.")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Run EDA
    data = explore_data()

    # Run preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(data)