# 12-Classification
KNN Classification Project: Diagnosing Cancer

# Business Objective

The primary objective of this project is to develop a machine learning model using the K-Nearest Neighbors (KNN) algorithm to classify whether a patient is likely to have benign or malignant cancer based on clinical data. This can aid healthcare professionals in making faster and more accurate diagnoses.

# Problem Statement

Cancer diagnosis is a critical task in healthcare, requiring accuracy and efficiency. Misdiagnosis can lead to improper treatment and severe consequences for patients. The challenge is to create a reliable system to classify cancer types based on diagnostic data.

# Solution

We implemented a KNN-based classification model that takes diagnostic data as input and predicts whether a tumor is benign or malignant. The solution involves data preprocessing, normalization, model training, evaluation, and optimization of the K value for improved accuracy.

# Dataset

The dataset consists of 369 rows and 32 columns, where each row represents a patient's diagnostic data.

The target variable is diagnosis, with two possible values:

B: Benign

M: Malignant

# Implementation Steps

1. Data Preprocessing

Remove unnecessary columns: Dropped the patient ID column as it does not contribute to the diagnosis.

Target variable transformation: Converted B to Benign and M to Malignant for better readability.

2. Normalization

Applied a normalization function to scale all feature columns to a range of 0 to 1.

3. Data Splitting

Split the dataset into training (80%) and testing (20%) sets using train_test_split.

Used stratified sampling to ensure balanced representation of target classes in both sets.

4. Model Training

Trained the KNN classifier with an initial value of k=21 and fitted it using the training data.

5. Model Evaluation

Predicted on the test set and evaluated the model using accuracy scores and confusion matrices.

Analyzed True Positives, True Negatives, False Positives, and False Negatives for interpretability.

6. Hyperparameter Tuning

Tested multiple values of k (from 3 to 50 in steps of 2) to identify the optimal k for maximum accuracy.

Plotted training and testing accuracy for different values of k.

# Key Results

Optimal value of k: 9

Accuracy of the model with k=9 on the test set: High accuracy was achieved, as indicated by the confusion matrix and accuracy score.

Code Highlights

Normalization function: Ensures that all features have equal weight during distance calculation.

Stratified sampling: Maintains the balance of classes in training and testing datasets.

Hyperparameter tuning: Iteratively tests different values of k to avoid overfitting or underfitting.

Visualization: Plots training and testing accuracy to evaluate the performance of the model.

# Technologies Used

Programming Language: Python

Libraries:

pandas for data manipulation

numpy for numerical operations

sklearn for model training, evaluation, and data splitting

matplotlib for data visualization

# How to Run the Project

Clone the repository.

Ensure the dataset wbcd.csv is placed in the appropriate directory.

Install the required Python libraries:

pip install pandas numpy scikit-learn matplotlib

Run the Python script containing the code.

View the results including the accuracy score, confusion matrix, and accuracy plots.

# Future Improvements

Incorporate feature selection techniques to reduce dimensionality.

Experiment with different distance metrics (e.g., Manhattan distance).

Apply other classification algorithms (e.g., SVM, Random Forest) for comparison.

Build a GUI for user-friendly interaction with the model.

# Conclusion

This project demonstrates the use of KNN for cancer diagnosis and highlights the importance of parameter tuning and proper data preprocessing in achieving high accuracy. The model serves as a proof of concept for machine learning applications in healthcare.

