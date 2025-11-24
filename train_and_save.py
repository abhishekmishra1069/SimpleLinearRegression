# train_and_save.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Your original code adapted for saving the model
def train_and_save():
    # Importing the dataset
    try:
        dataset = pd.read_csv('Salary_Data.csv')
    except FileNotFoundError:
        print("Error: Salary_Data.csv not found. Please place it in the same directory.")
        return

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset is not strictly necessary for final training,
    # but we'll keep the fit process consistent with your code for robustness.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    # Training the Simple Linear Regression model on the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Save the trained model to disk
    with open('model.pkl', 'wb') as file:
        pickle.dump(regressor, file)

    print("Model trained and saved to model.pkl")

if __name__ == "__main__":
    train_and_save()