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
    # 1. Load the Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # 2. Split the data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 3. Train a classification model (MLPClassifier allows loss tracking)
    # We use a Multi-layer Perceptron (Neural Network) for classification
    clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42, solver='adam')
    clf.fit(X_train, y_train)

    # 4. Generate a convergence graph (Loss over iterations)
    plt.figure(figsize=(10, 6))
    plt.plot(clf.loss_curve_)
    plt.title('Convergence Graph: Training Loss over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('convergence_graph.png')
    plt.close()
    print("Convergence graph saved as 'convergence_graph.png'")

    # 5. Evaluate the model and Generate a confusion matrix
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix (Accuracy: {acc:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

    # 6. Generate an additional insightful chart (Feature Scatter Plot)
    # Visualizing Petal Length vs Petal Width as they are usually the best discriminators
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species', style='species', s=100)
    plt.title('Iris Feature Insight: Petal Length vs Petal Width')
    plt.grid(True)
    plt.savefig('extra_chart.png')
    plt.close()
    print("Extra chart saved as 'extra_chart.png'")

    # 7. Print Summary Metrics
    print("\nModel Summary Report:")
    print("-" * 30)
    print(f"Final Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    run_iris_classification()
