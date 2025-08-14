

# Python Learning Projects

This repository contains a collection of machine learning projects implemented in Python using Jupyter Notebooks. Each project explores a different dataset and machine learning technique.

-----

## Table of Contents

1.  [Titanic Survival Prediction](https://www.google.com/search?q=%231-titanic-survival-prediction-)
2.  [Sales Prediction Using Advertising Data](https://www.google.com/search?q=%232-sales-prediction-using-advertising-data-)

-----

# 1\. Titanic Survival Prediction 🚢

This project aims to predict the survival of passengers aboard the RMS Titanic using a machine learning model. It uses passenger data like ticket class and gender to train a Logistic Regression classifier to determine if a passenger survived the disaster.

### Dataset 📊

This project uses the Titanic dataset, which contains information about the passengers. The model is built using the following features:

  * **`Pclass`**: The passenger's ticket class (1, 2, or 3).
  * **`Sex`**: The passenger's gender (male or female).

The target variable to be predicted is:

  * **`Survived`**: A binary value indicating survival (1 for Survived, 0 for Not Survived).

-----

### ⚙️ Project Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading and Exploration**: The dataset (`tested.csv`) is loaded, and an initial analysis is performed to understand its structure.
2.  **Data Visualization**: Relationships between features like `Pclass` and `Sex` versus the `Survived` outcome are visualized using `seaborn`.
3.  **Data Preprocessing**:
      * The categorical `Sex` column is converted to numerical values (male: 1, female: 0) using `LabelEncoder`.
      * Missing values are addressed by dropping the `Age` column, which contained many null entries.
4.  **Model Training**:
      * Features (`X`) are selected (`Pclass`, `Sex`).
      * The target (`Y`) is set to the `Survived` column.
      * The data is split into training and testing sets (80-20 split).
      * A **Logistic Regression** model is trained on the training data.
5.  **Prediction**: The trained model is used to make predictions on the test set and on new, unseen data.

-----

### 🛠️ Libraries Required

To run this project, you'll need the following Python libraries:

  * `numpy`
  * `pandas`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn`

-----

### 🚀 How to Run

1.  **Clone the repository or download the project files.**

2.  **Install the necessary libraries** by running the following command in your terminal:

    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

3.  **Run the Jupyter Notebook** `TITANIC_SURVIVAL_PREDICTION (1).ipynb` in your preferred environment (like Jupyter Lab or VS Code). Execute the cells sequentially to see the analysis and results.

-----

### ✅ Results

The model is trained to predict whether a passenger survived. The final section of the notebook allows you to input a passenger's class and sex to receive a prediction.

For example, to predict the survival of a male passenger in 2nd class:

```python
# The input [2, 1] represents Pclass=2 and Sex=1 (male)
res = log.predict([[2, 1]])

if(res == 0):
  print("So Sorry! Not Survived")
else:
  print("Survived")
```

-----

# 2\. Sales Prediction Using Advertising Data 📈

This project analyzes an advertising dataset to build a machine learning model that predicts sales based on the amount of money spent on different advertising platforms. A Simple Linear Regression model is implemented to understand the relationship between TV advertising expenditure and sales.

### 💾 Dataset

The project utilizes the **Advertising dataset** (`advertising.csv`), which contains data on advertising spending and corresponding sales.

The features in the dataset are:

  * **TV**: Advertising budget spent on TV (in thousands of dollars).
  * **Radio**: Advertising budget spent on Radio (in thousands of dollars).
  * **Newspaper**: Advertising budget spent on Newspaper (in thousands of dollars).

The target variable is:

  * **Sales**: Product sales (in thousands of units).

-----

### ⚙️ Workflow

The project follows these key steps:

1.  **Data Loading & Exploration**: The `advertising.csv` dataset is loaded using pandas. Initial exploration is performed to check its shape, and descriptive statistics.

2.  **Exploratory Data Analysis (EDA)**:

      * A pairplot and histograms are created to visualize the relationships between advertising channels and sales.
      * A correlation heatmap is generated, which reveals a strong positive correlation (**0.9**) between TV advertising and Sales, making it the best predictor for a simple linear model.

3.  **Model Training**:

      * The dataset is split into features (`X`) and the target variable (`y`). For this simple regression model, **'TV'** is chosen as the sole feature.
      * The data is divided into a training set (70%) and a testing set (30%).
      * A **Simple Linear Regression** model from scikit-learn is trained on the training data.

4.  **Evaluation & Prediction**:

      * The trained model is used to make predictions on the test dataset.
      * The model's coefficient and intercept are extracted to form the linear regression equation.
      * A scatter plot of the test data is created with the regression line overlaid to visually assess the model's performance.

-----

### 🛠️ Technologies and Libraries Used

  * `Python`
  * `Pandas`
  * `NumPy`
  * `Matplotlib` & `Seaborn`
  * `Scikit-learn`
  * `Jupyter Notebook`

-----

### 🚀 Setup and Usage

To run this project on your local machine, follow these steps:

1.  **Clone the repository or download the project files.**

2.  **Install the necessary libraries** by running the following command in your terminal:

    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

3.  **Run the Jupyter Notebook** `sales_prediction.ipynb` in your preferred environment. Execute the cells to see the complete analysis, model training, and results.

-----

### ✅ Results

The Linear Regression model successfully captures the relationship between TV advertising and sales. The final model can be represented by the equation:

**`Sales ≈ 7.14 + 0.055 * TV`**

This equation indicates that for every additional $1000 spent on TV advertising, sales are predicted to increase by approximately 55 units. The final visualization shows the regression line fitting well with the test data, confirming the model's predictive capability.

-----

### ✍️ Author

  * **ABHINANDAN KESARWANI**
