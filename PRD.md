# Product Requirements Document (PRD): Iris Classification Project

## 1. Objective
The primary goal of this project is to build a machine learning classification model to accurately categorize Iris flowers into one of three species based on morphological measurements.

*   **Model Split:** The dataset will be partitioned into a **Train set (80%)** and a **Test set (20%)** to ensure robust evaluation and prevent overfitting.
*   **Success Metric:** High accuracy on the test set and a clear interpretation of model performance via a confusion matrix.

## 2. Dataset Definition
The [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) is a classic multiclass classification dataset.

### 2.1 Features (Inputs)
The model will use four numerical features (measured in cm):
1.  **Sepal Length:** Length of the sepal.
2.  **Sepal Width:** Width of the sepal.
3.  **Petal Length:** Length of the petal.
4.  **Petal Width:** Width of the petal.

### 2.2 Target Categories (Outputs)
The model will classify each observation into one of three species:
1.  **Iris Setosa**
2.  **Iris Versicolour**
3.  **Iris Virginica**

## 3. Required Deliverables
To complete the project, the following artifacts must be provided:

*   **Python Code:** A clean, documented script (`iris_classifier.py`) that handles data loading, preprocessing, training, and evaluation.
*   **Confusion Matrix:** A visualization (e.g., a heatmap) showing the model's true positive vs. false positive predictions for each category.
*   **Convergence Graph:** A plot showing the **Loss over Iterations** (or Accuracy over Epochs) to demonstrate the model's learning process.
*   **Summary Report:** A brief analysis of the results, including final accuracy and any insights gained from the model's performance.

## 4. Technical Constraints
*   **Language:** Python 3.x
*   **Libraries:** `scikit-learn`, `pandas`, `matplotlib`, `seaborn`.
