import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def train_models():
    """Train Logistic Regression, Decision Tree, and k-NN models."""
    # Load preprocessed data
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/y_test.csv').values.ravel()

    # Initialize models
    log_reg = LogisticRegression(random_state=42)
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train models
    log_reg.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    return log_reg, dt, knn, X_test, y_test

def evaluate_models():
    """Evaluate trained models and visualize confusion matrices."""
    # Train models and get test data
    log_reg, dt, knn, X_test, y_test = train_models()

    # Make predictions
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_dt = dt.predict(X_test)
    y_pred_knn = knn.predict(X_test)

    # Dictionary of models and predictions
    models = {
        'Logistic Regression': y_pred_log_reg,
        'Decision Tree': y_pred_dt,
        'k-NN': y_pred_knn
    }

    # Evaluate each model
    for name, y_pred in models.items():
        print(f"\n{name} Performance:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # Visualize confusion matrix
        plt.figure(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title(f'Confusion Matrix: {name}')
        plt.savefig(f'../results/confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()

if __name__ == "__main__":
    # Evaluate models
    evaluate_models()