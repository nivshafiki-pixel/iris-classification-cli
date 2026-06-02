# Project Report: Iris Classification with 4 Categories

## Executive Summary
This project successfully developed a Machine Learning model capable of classifying Iris flowers into **four** distinct categories, including a synthetic 4th class ("Iris Synthetic") created to increase model complexity.

## Methodology
- **Data Augmentation:** Added 50 synthetic samples with specific feature distributions to the original 150-sample Iris dataset.
- **Persistence:** The full 200-sample dataset was exported to `iris_dataset.csv`.
- **Training:** Used a Multi-layer Perceptron (MLP) with an 80/20 train-test split.
- **Optimization:** Trained for up to 2,000 iterations to ensure convergence on the expanded dataset.

## Results
- **Final Accuracy:** 95% on the test set.
- **Confusion Matrix:** Successfully generated as a 4x4 matrix, showing high precision across all species.
- **Convergence:** The loss curve shows stable learning, as seen in `convergence_graph.png`.

## Artifacts
1. `iris_dataset.csv` - The raw data.
2. `confusion_matrix.png` - 4x4 performance heatmap.
3. `convergence_graph.png` - Loss over iterations plot.
4. `extra_chart.png` - Feature distribution scatter plot.
