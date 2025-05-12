import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data():
    """Load preprocessed data."""
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def train_and_evaluate():
    """Train models and compute accuracies."""
    X_train, X_test, y_train, y_test = load_data()

    # Initialize and train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'k-NN': KNeighborsClassifier(n_neighbors=5)
    }
    accuracies = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)

    return accuracies

def visualize_results():
    """Visualize model accuracy comparison."""
    accuracies = train_and_evaluate()

    # Create bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('../results/model_accuracy_comparison.png')
    plt.close()

    print("\nModel Accuracies:")
    for name, acc in accuracies.items():
        print(f"{name}: {acc:.4f}")
    print("Accuracy comparison plot saved to results/model_accuracy_comparison.png")

if __name__ == "__main__":
    visualize_results()