Car Price Prediction using Machine Learning
CodeAlpha Data Science Internship – Task 3
* Project Overview

This project is part of the CodeAlpha Data Science Internship.
It predicts car selling prices using Machine Learning techniques.

The project includes feature engineering, model training, evaluation, and saving the trained model for future predictions.

* Project Objective

To build a machine learning model that predicts car selling prices based on features such as:

Present Price

Kilometers Driven

Fuel Type

Transmission Type

Owner Type

Car Age

* Dataset

Used Car Price Prediction Dataset (CSV format)

Features:

Present_Price

Kms_Driven

Fuel_Type

Seller_Type

Transmission

Owner

Year

Target:

Selling_Price

* Algorithm Used

Random Forest Regressor (Advanced Model)

* Tools & Libraries

Python

Jupyter Notebook

Pandas

NumPy

Scikit-learn

Matplotlib

Pickle (for model saving)

* Project Workflow

Load and inspect dataset

Data cleaning and preprocessing

Feature engineering (Car Age creation)

Encode categorical variables

Train-test split

Train Random Forest model

Evaluate using MAE, MSE, R² score

Perform cross-validation

Analyze feature importance

Save trained model using pickle

*Model Performance

Evaluated using:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

R² Score

Cross-validation Score

The model achieved strong predictive performance on unseen test data.

* Model Saving

The trained model and scaler are saved as:

car_price_model.pkl

scaler.pkl

These files allow predictions without retraining the model.

* How to Run
Train Model
python train_model.py

Predict Using Saved Model
python predict.py

    CHAITANYA VIKRAM RATHOD

CodeAlpha Data Science Intern