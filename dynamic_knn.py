from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def distance_manhattan(x1, x2):
    return np.sum(np.abs(x1 - x2))


def distance_euclidean(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class DynamicKNN:
    def __init__(self, max_neighbors=10, max_spread=0.5):
        self.max_neighbors = max_neighbors
        self.max_spread = max_spread
        self.scaler = MinMaxScaler()

    def fit(self, X: np.array, y: np.array):
        self.X = self.scaler.fit_transform(X)
        self.y = y

    def predict(self, X_test: np.array):
        X_test = self.scaler.transform(X_test)
        predictions = []

        for test_point in X_test:
            dist = [distance_manhattan(train_point, test_point) for train_point in self.X]
            index_list = np.argsort(dist)

            selected = []
            for i in index_list[:self.max_neighbors]:
                selected.append(i)
                if len(selected) > 1:
                    spread = np.ptp([dist[j] for j in selected])
                    if spread > self.max_spread:
                        selected.pop()
                        break

            y_list = [self.y[i] for i in selected]
            most_common = Counter(y_list).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)


class BaseKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X: np.array, y: np.array):
        self.X = X
        self.y = y

    def predict(self, X_test: np.array):
        predictions = []

        for test_point in X_test:
            dist = [distance_euclidean(train_point, test_point) for train_point in self.X]
            index_list = np.argsort(dist)
            sliced = index_list[0:self.n_neighbors]

            y_list = [self.y[i] for i in sliced]
            most_common = Counter(y_list).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)


def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    dynamic_model = DynamicKNN(max_neighbors=10, max_spread=0.5)
    base_model = BaseKNN(n_neighbors=5)

    dynamic_model.fit(X_train, y_train)
    base_model.fit(X_train, y_train)

    dyn_preds = dynamic_model.predict(X_test)
    base_preds = base_model.predict(X_test)

    print("== Dynamic KNN ==")
    print(f"Accuracy: {accuracy_score(y_test, dyn_preds):.2f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, dyn_preds))
    print()

    print("== Base KNN ==")
    print(f"Accuracy: {accuracy_score(y_test, base_preds):.2f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, base_preds))

    plot_conf_matrix(y_test, dyn_preds, "Dynamic KNN - Confusion Matrix")
    plot_conf_matrix(y_test, base_preds, "Base KNN - Confusion Matrix")

    accuracy_data = pd.DataFrame({
        'Model': ['Dynamic KNN', 'Base KNN'],
        'Accuracy': [
            accuracy_score(y_test, dyn_preds),
            accuracy_score(y_test, base_preds)
        ]
    })

    plt.figure(figsize=(5, 3))
    sns.barplot(data=accuracy_data, x='Model', y='Accuracy')
    plt.ylim(0.9, 1.0)
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.show()