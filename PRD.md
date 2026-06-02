# Product Requirements Document (PRD): Iris Classification Project (Expanded)

## 1. Objective
The primary goal of this project is to build a machine learning classification model to accurately categorize Iris flowers into one of **four** species based on morphological measurements. This version introduces a synthetic 4th category to increase the model's complexity.

*   **Model Split:** The dataset is partitioned into a **Train set (80%)** and a **Test set (20%)**.
*   **Success Metric:** High accuracy on the test set and a clear interpretation of model performance via a **4x4 confusion matrix**.

## 2. Dataset Definition
The dataset consists of the original [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) plus a synthetic 4th category. The complete dataset is saved locally as **`iris_dataset.csv`**.

### 2.1 Features (Inputs)
The model uses four numerical features (measured in cm):
1.  **Sepal Length**
2.  **Sepal Width**
3.  **Petal Length**
4.  **Petal Width**

### 2.2 Target Categories (Outputs)
The model classifies each observation into one of four species:
1.  **Iris Setosa**
2.  **Iris Versicolour**
3.  **Iris Virginica**
4.  **Iris Synthetic** (Added for increased complexity)

## 3. Required Deliverables
*   **Python Code:** Updated script (`iris_machine.py`) that generates the 4th category, saves the CSV, and trains the model.
*   **Local CSV Dataset:** `iris_dataset.csv` containing all 4 categories.
*   **Confusion Matrix:** A **4x4 heatmap** visualization showing the model's predictions.
*   **Convergence Graph:** A plot showing the **Loss over Iterations** for the 4-class training cycle.
*   **Summary Report:** Detailed analysis in Markdown (English and Hebrew).

## 4. Technical Constraints
*   **Libraries:** `scikit-learn`, `pandas`, `matplotlib`, `seaborn`.
