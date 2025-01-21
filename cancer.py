# **Business Objective**
# The goal is to classify tumors as Malignant (cancerous) or Benign (non-cancerous)
# using the K-Nearest Neighbors (KNN) classification algorithm based on features derived from tumor samples.

# **Problem Statement**
# Given a dataset containing tumor features, the task is to develop a model that can accurately classify 
# tumors as Malignant or Benign. The challenge is to identify the best value of k for the KNN model 
# and ensure high accuracy in predictions.

# **Solution**
# 1. Load and preprocess the data.
# 2. Train a KNN classifier on the training data.
# 3. Tune the k hyperparameter to find the optimal number of neighbors.
# 4. Evaluate the model's performance and choose the best configuration.

import pandas as pd
import numpy as np

# **Step 1: Load the data**
# Load the dataset that contains features of tumors and their diagnosis
wbcd = pd.read_csv("E:/Data Science/12-Classification/wbcd.csv")  # Replace with the actual path

# Display summary statistics to understand the data
wbcd.describe()  # Provides an overview of numerical features in the dataset

# **Step 2: Preprocess the data**
# Replace 'B' with 'Benign' and 'M' with 'Malignant' for better readability
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'B', 'Benign', wbcd['diagnosis'])
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'M', 'Malignant', wbcd['diagnosis'])

# Drop the patient ID column as it is not relevant to the classification
wbcd = wbcd.iloc[:, 1:32]  # Selecting all columns except the first one (patient ID)

# **Step 3: Normalize the data**
# Define a normalization function to scale all features between 0 and 1
def norm_func(i):
    return (i - i.min()) / (i.max() - i.min())  # Formula for min-max normalization

# Normalize all feature columns (excluding the diagnosis column)
wbcd_n = norm_func(wbcd.iloc[:, 1:])  # Normalize only numerical features

# Define the target variable (diagnosis) for predictions
y = np.array(wbcd['diagnosis'])  # Target labels: Malignant or Benign

# **Step 4: Split the dataset**
# Split the dataset into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wbcd_n, y, test_size=0.2, random_state=42)

# **Step 5: Train the KNN model**
# Initialize the KNN classifier with k=21 (default value to start)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)  # Set k=21
knn.fit(X_train, y_train)  # Train the KNN classifier on the training data

# Make predictions on the test data
pred = knn.predict(X_test)

# **Step 6: Evaluate the model**
# Calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
print(f"Accuracy with k=21: {accuracy_score(pred, y_test)}")  # Display accuracy

# Generate a confusion matrix to analyze the performance
conf_matrix = pd.crosstab(pred, y_test)
print(conf_matrix)  # Shows true positives, true negatives, false positives, false negatives

# **Step 7: Hyperparameter tuning**
# Experiment with different k-values to find the optimal number of neighbors
# Create an empty list to store training and testing accuracies for each k
acc = []

# Loop through odd values of k from 3 to 49
for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors=i)  # Set k=i
    neigh.fit(X_train, y_train)  # Train the model
    train_acc = np.mean(neigh.predict(X_train) == y_train)  # Training accuracy
    test_acc = np.mean(neigh.predict(X_test) == y_test)  # Testing accuracy
    acc.append([train_acc, test_acc])  # Append both accuracies to the list

# **Step 8: Plot accuracy vs k-values**
import matplotlib.pyplot as plt

# Plot training accuracy
plt.plot(np.arange(3, 50, 2), [i[0] for i in acc], "ro-", label="Training Accuracy")
# Plot testing accuracy
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "bo-", label="Testing Accuracy")

# Add labels, title, and legend to the plot
plt.xlabel("k-value (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Testing Accuracy for Different k-values")
plt.show()

# **Step 9: Final Model with Optimal k**
# Based on the plot, select the k-value with the highest testing accuracy (e.g., k=9)
knn = KNeighborsClassifier(n_neighbors=9)  # Optimal k-value
knn.fit(X_train, y_train)  # Retrain the model with the optimal k

# Make predictions on the test data with the final model
pred = knn.predict(X_test)

# Evaluate the final model
final_accuracy = accuracy_score(pred, y_test)
print(f"Final Accuracy with k=9: {final_accuracy}")  # Print the final accuracy
conf_matrix_final = pd.crosstab(pred, y_test)
print(conf_matrix_final)  # Print the confusion matrix for the final model
