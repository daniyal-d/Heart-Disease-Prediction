# Heart-Disease-Prediction
Streamlit web app that uses a KNN classification model to predict whether or not someone has heart disease. It has 92% accuracy, .92 recall, and a .92 F1-score.

Website link: 

This project uses the University of California Irvine's heart disease dataset (https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) to train and evaluate predictions. This repository consists a Jupyter Notebook, which contains analysis of the data, model training/hypertuning, and model evaluation, the saved classification model, and a Python file containing the code of the web app.

# How the model works
This project uses a machine learning classifier model (KNN) to predict heart disease. The dataset contains multiple different variables about the health of patients. These values are all either quantitative, or categorical but encoded as numeric values (for example, in the sex variable, 1 = Male and 0 = Female). This dataset was tested on 3 different classifiers (Linear Regression, KNN, and Random Forest) and baseline results showed that logistic regression had the highest accuracy. However, after standerdizing the data with `StandardScaler()`, KNN had the highest accuracy. 

# $ z = \dfrac{x-μ}{σ} $

Standardization works by taking the mean (μ) and standard deviation (σ) of each column. Then, each individual value's z-score (standardized value) is calculated

# Hypertuning
Hyptertuning was not able to increase the accuracy scores of Logistic Regression or Random Forest on neither the scaled or unscaled data, but it pushed KNN's accuracy from ~90% to ~92%. I also attempted to encode the categorical variables with `OneHotEncoder` and `pd.get_dummies`, and then scaling the quantitative variables (not shown in the notebook), but this did not improve any model's score. Overall, I found that a KNN classifier with n_neighbors=7 is the most accurate model. This is what the website uses.
