import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os

# Set the working directory to the current file's location
os.chdir(os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else '.')

def run_iris_classification():
    # 1. Load the Dataset and Add a Synthetic 4th Category
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = list(iris.target_names) + ['iris synthetic']

    # Create 50 synthetic samples (e.g., larger dimensions to differentiate)
    # Mean values are slightly higher than Virginica's to create a distinct 4th class
    synthetic_X = np.random.normal(loc=[7.5, 3.5, 6.5, 2.5], scale=0.2, size=(50, 4))
    synthetic_y = np.full(50, 3)

    X_combined = np.vstack([X, synthetic_X])
    y_combined = np.concatenate([y, synthetic_y])

    # 2. Save the updated dataset locally as 'iris_dataset.csv'
    df_full = pd.DataFrame(X_combined, columns=feature_names)
    df_full['target'] = y_combined
    df_full['species'] = [target_names[i] for i in y_combined]
    df_full.to_csv('iris_dataset.csv', index=False)
    print("Dataset saved as 'iris_dataset.csv'")

    # 3. Split the data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.20, random_state=42)

    # 4. Train a classification model
    clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000, random_state=42, solver='adam')
    clf.fit(X_train, y_train)

    # 5. Generate a convergence graph (Loss over iterations)
    plt.figure(figsize=(10, 6))
    plt.plot(clf.loss_curve_)
    plt.title('Convergence Graph: Training Loss over Iterations (4 Classes)')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('convergence_graph.png')
    plt.close()
    print("Convergence graph saved as 'convergence_graph.png'")

    # 6. Evaluate the model and Generate a 4x4 confusion matrix
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix (4x4) - Accuracy: {acc:.2f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("4x4 Confusion matrix saved as 'confusion_matrix.png'")

    # 7. Generate extra chart (Feature Scatter Plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_full, x='petal length (cm)', y='petal width (cm)', hue='species', style='species', s=100)
    plt.title('Iris Feature Insight: 4 Categories Comparison')
    plt.grid(True)
    plt.savefig('extra_chart.png')
    plt.close()
    print("Extra chart saved as 'extra_chart.png'")

    # 8. Print Summary Metrics
    print("\nModel Summary Report (4 Classes):")
    print("-" * 35)
    print(f"Final Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    run_iris_classification()
