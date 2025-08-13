# Python Learning Projects 


## Titanic Survival Prediction

This project uses machine learning to predict whether a passenger on the RMS Titanic survived the infamous 1912 disaster. The prediction is based on passenger data, and the model is built using a Logistic Regression algorithm.

📋 Table of Contents
Project Overview

Dataset

Workflow

Technologies and Libraries Used

Setup and Usage

Results

🔭 Project Overview
The goal of this project is to analyze the Titanic passenger dataset and build a predictive model to determine a passenger's likelihood of survival. The analysis involves data exploration, visualization, preprocessing, and finally, training a machine learning model.

💾 Dataset
The project utilizes the Titanic dataset, which is a well-known dataset in the machine learning community. The tested.csv file contains various details for 418 passengers.

The key features used for prediction in this model are:

Pclass: The passenger's ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).

Sex: The passenger's gender.

The target variable is:

Survived: Indicates if the passenger survived (1 = Yes, 0 = No).

⚙️ Workflow
The project follows these steps:

Data Loading & Exploration: The tested.csv dataset is loaded into a pandas DataFrame. Initial exploration is done to understand the data's structure and summary statistics.

Exploratory Data Analysis (EDA): Visualizations are created using Seaborn to find relationships between different features and the survival outcome. Key findings include:

Female passengers had a significantly higher survival rate than male passengers.

Passengers in higher classes (1st class) had a better chance of survival.

Data Preprocessing:

The categorical Sex feature is converted into a numerical format using LabelEncoder (female: 0, male: 1).

Missing values are handled. The Age column, which has a significant number of missing values, is dropped to simplify the model.

Model Training:

The dataset is split into features (X) consisting of Pclass and Sex, and the target variable (Y) which is Survived.

The data is further divided into training and testing sets.

A Logistic Regression model is trained on the training data.

Prediction: The trained model is used to make survival predictions on the test dataset and on custom input.

🛠️ Technologies and Libraries Used
Python

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Matplotlib & Seaborn: For data visualization.

Scikit-learn: For machine learning tasks, including:

LogisticRegression for the classification model.

train_test_split for splitting the dataset.

LabelEncoder for data preprocessing.

Jupyter Notebook: For interactive development and analysis.

🚀 Setup and Usage
To run this project on your local machine, follow these steps:

Clone the repository:

Bash

git clone <your-repository-url>
Navigate to the project directory:

Bash

cd <project-directory>
Install the required libraries:

Bash

pip install numpy pandas matplotlib seaborn scikit-learn
Run the Jupyter Notebook:

Bash

jupyter notebook "TITANIC_SURVIVAL_PREDICTION (1).ipynb"
You can then execute the cells to see the analysis and model predictions.

📈 Results
The trained Logistic Regression model predicts whether a passenger survived or not. The final cell in the notebook provides an example of how to use the model for a new prediction:

Python

# Predicts the outcome for a male passenger in 2nd class
res = log.predict([[2, 1]])

if(res == 0):
  print("So Sorry! Not Survived")
else:
  print("Survived")
This demonstrates the model's ability to classify passengers based on the selected features. The model's performance shows a clear correlation where female passengers had a 100% survival rate in the test data, while male passengers did not.


