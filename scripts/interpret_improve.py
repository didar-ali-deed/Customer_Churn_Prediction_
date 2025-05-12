import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """Load preprocessed data."""
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def feature_importance():
    """Analyze feature importance for Decision Tree."""
    X_train, X_test, y_train, y_test = load_data()
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)

    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': dt.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features (Decision Tree):")
    print(feature_importance.head(10))

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
    plt.title('Top 10 Feature Importance (Decision Tree)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png')
    plt.close()

def tune_knn():
    """Tune k-NN hyperparameters using Grid Search."""
    X_train, X_test, y_train, y_test = load_data()
    param_grid = {'n_neighbors': [3, 5, 7, 9]}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("\nGrid Search Results for k-NN:")
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # Evaluate tuned k-NN on test set
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    print("\nTuned k-NN Performance on Test Set:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def apply_smote():
    """Apply SMOTE to handle class imbalance and retrain k-NN."""
    X_train, X_test, y_train, y_test = load_data()
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("\nSMOTE Applied:")
    print("Original y_train distribution:", pd.Series(y_train).value_counts().to_dict())
    print("Balanced y_train distribution:", pd.Series(y_train_bal).value_counts().to_dict())

    # Train k-NN on balanced data
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_bal, y_train_bal)
    y_pred = knn.predict(X_test)

    print("\nk-NN Performance on Test Set (with SMOTE):")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Run interpretation and improvement tasks
    feature_importance()
    tune_knn()
    apply_smote()