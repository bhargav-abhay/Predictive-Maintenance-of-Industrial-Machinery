# EDUNET FOUNDATION MODEL 
                                                     
# Predictive Maintenance ML Model Comparison
This repository contains Python code for analyzing a predictive maintenance dataset, training various machine learning models to predict equipment failures, and comparing their accuracy.

# 📊 Project Overview
The goal of this project is to build and evaluate machine learning models that can predict different types of equipment failures based on sensor readings and operational parameters. By accurately predicting failures, maintenance can be scheduled proactively, reducing downtime and operational costs.

 #📁 Dataset
The analysis uses the predictive_maintenance.csv dataset. This dataset typically includes:

UDI: Unique identifier for the data point.

Product ID: Identifier for the product.

Type: Type of machine (e.g., Low, Medium, High).

Air temperature [K]: Air temperature in Kelvin.

Process temperature [K]: Process temperature in Kelvin.

Rotational speed [rpm]: Rotational speed in revolutions per minute.

Torque [Nm]: Torque in Newton-meters.

Tool wear [min]: Tool wear in minutes.

Target: Binary indicator of whether a failure occurred (0 or 1).

Failure Type: Categorical type of failure (e.g., 'No Failure', 'Power Failure', 'Tool Wear Failure', etc.).

# 🛠️ Methodology
The ibmcloudmyapproach.py script performs the following steps:

Data Loading: Loads the predictive_maintenance.csv dataset using Pandas.

 # Preprocessing:

Drops irrelevant columns (UDI, Product ID, Target).

Encodes the Failure Type target variable into numerical labels using LabelEncoder.

Performs one-hot encoding on the Type categorical feature.

Splits the data into training and testing sets (70% train, 30% test) while maintaining the original class distribution (stratify=y).

Scales numerical features using StandardScaler to ensure all features contribute equally to the model training.

Model Training: Trains three different classification models:

Logistic Regression: A linear model for binary/multiclass classification.

Random Forest Classifier: An ensemble learning method that builds multiple decision trees.

Gradient Boosting Classifier: Another ensemble method that builds trees sequentially, correcting errors of previous trees.

Model Evaluation: Calculates and prints the accuracy score for each model on the test set.

Visualization: Generates a bar chart using Altair to visually compare the accuracy of the models, saving it as model_accuracy_comparison.json.

# 📈 Results
The script outputs the accuracy scores for each trained model. In a typical run, you might see results similar to:

Accuracy Scores:
Logistic Regression: 0.9687
Random Forest: 0.9800
Gradient Boosting: 0.9800

This indicates that both Random Forest and Gradient Boosting models achieved the highest accuracy (98%) in predicting Failure Type, slightly outperforming Logistic Regression.

A visualization of these accuracies is saved as model_accuracy_comparison.json (which can be viewed in a Vega-Lite compatible viewer).



# Prerequisites
Python 3.x installed

Pip (Python package installer)


Install the required Python libraries:

pip install pandas scikit-learn altair

# Execution
Place your predictive_maintenance.csv file in the same directory as the ibmcloudmyapproach.py script.

Run the Python script from your terminal:

python ibmcloudmyapproach.py

The script will print the accuracy scores to the console and generate a model_accuracy_comparison.json file in the same directory, which contains the data for the comparison chart.

# 🤝 Contributing
Feel free to fork this repository, open issues, or submit pull requests.


