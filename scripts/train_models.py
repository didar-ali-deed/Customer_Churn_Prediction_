import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_models():
    """Train Logistic Regression, Decision Tree, and k-NN models."""
    # Load preprocessed data
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/y_test.csv').values.ravel()

    # Initialize models
    log_reg = LogisticRegression(random_state=42)
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)  # Limit depth to avoid overfitting
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train models
    log_reg.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    print("\nModels Trained:")
    print("Logistic Regression: Trained")
    print("Decision Tree: Trained (max_depth=5)")
    print("k-NN: Trained (n_neighbors=5)")

    return log_reg, dt, knn, X_test, y_test

if __name__ == "__main__":
    # Train models
    log_reg, dt, knn, X_test, y_test = train_models()