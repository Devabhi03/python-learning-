# Python Learning Projects 


# Titanic Survival Prediction 🚢

This project aims to predict the survival of passengers aboard the RMS Titanic using a machine learning model. It uses passenger data like ticket class and gender to train a Logistic Regression classifier to determine if a passenger survived the disaster.

---

##  Dataset 📊

This project uses the Titanic dataset, which contains information about the passengers. The model is built using the following features:

-   **`Pclass`**: The passenger's ticket class (1, 2, or 3).
-   **`Sex`**: The passenger's gender (male or female).

The target variable to be predicted is:

-   **`Survived`**: A binary value indicating survival (1 for Survived, 0 for Not Survived).

---

## ⚙️ Project Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading and Exploration**: The dataset (`tested.csv`) is loaded, and an initial analysis is performed to understand its structure.
2.  **Data Visualization**: Relationships between features like `Pclass` and `Sex` versus the `Survived` outcome are visualized using `seaborn`.
3.  **Data Preprocessing**:
    -   The categorical `Sex` column is converted to numerical values (male: 1, female: 0) using `LabelEncoder`.
    -   Missing values are addressed by dropping the `Age` column, which contained many null entries.
4.  **Model Training**:
    -   Features (`X`) are selected (`Pclass`, `Sex`).
    -   The target (`Y`) is set to the `Survived` column.
    -   The data is split into training and testing sets (80-20 split).
    -   A **Logistic Regression** model is trained on the training data.
5.  **Prediction**: The trained model is used to make predictions on the test set and on new, unseen data.

---

## 🛠️ Libraries Required

To run this project, you'll need the following Python libraries:

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`

---

## 🚀 How to Run

1.  **Clone the repository or download the project files.**

2.  **Install the necessary libraries** by running the following command in your terminal:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

3.  **Run the Jupyter Notebook** `TITANIC_SURVIVAL_PREDICTION (1).ipynb` in your preferred environment (like Jupyter Lab or VS Code). Execute the cells sequentially to see the analysis and results.

---

## ✅ Results

The model is trained to predict whether a passenger survived. The final section of the notebook allows you to input a passenger's class and sex to receive a prediction.

For example, to predict the survival of a male passenger in 2nd class:

```python
# The input [2, 1] represents Pclass=2 and Sex=1 (male)
res = log.predict([[2, 1]])

if(res == 0):
  print("So Sorry! Not Survived")
else:
  print("Survived")



