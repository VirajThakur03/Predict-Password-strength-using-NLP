# Predict-Password-strength-using-NLP
Password Strength Detector
This project implements a machine learning model to classify the strength of passwords into three categories: Weak (0), Medium (1), and Strong (2). The model uses various features extracted from the passwords, including length, frequency of lowercase, uppercase, numeric, and special characters, and character-level TF-IDF representations.

Table of Contents
Project Overview
Data
Feature Engineering
Model Training
Prediction
Evaluation
Usage
Project Overview
The goal of this project is to build a robust password strength classification system. This system can be used to help users create stronger passwords or to analyze the overall strength of a set of passwords.

Data
The dataset password_data.sqlite contains password strings and their corresponding strength labels (0, 1, or 2).

Data Loading
The data is loaded from an SQLite database:

import sqlite3
import pandas as pd

sq = sqlite3.connect(r"/content/drive/MyDrive/Colab Notebooks/password_data.sqlite")
data = pd.read_sql_query("SELECT * FROM Users", sq)
data.drop(["index"], axis=1, inplace=True)
Feature Engineering
Several features are engineered from the raw password strings:

Length: The total number of characters in the password.
Lowercase Frequency: The proportion of lowercase characters.
Uppercase Frequency: The proportion of uppercase characters.
Number Frequency: The proportion of numeric characters.
Special Character Frequency: The proportion of special characters (non-alphanumeric).
TF-IDF Vectorization: Character-level Term Frequency-Inverse Document Frequency (TF-IDF) is applied to represent passwords as numerical vectors, capturing the importance of individual characters.
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def frequency_lower(row):
  return len([char for char in row if char.islower()]) / len(row)

def frequency_upper(row):
  return len([char for char in row if char.isupper()]) / len(row)

def frequency_number(row):
  return len([char for char in row if char.isnumeric()]) / len(row)

def frequency_special(row):
  special = []
  for char in row:
    if not char.isdigit() and not char.isalpha():
      special.append(char)
  return len(special) / len(row)

data["length"] = data["password"].str.len()
data["lower_frequency"] = np.round(data["password"].apply(frequency_lower), 3)
data["upper_frequency"] = np.round(data["password"].apply(frequency_upper), 3)
data["number_frequency"] = np.round(data["password"].apply(frequency_number), 3)
data["special_frequency"] = np.round(data["password"].apply(frequency_special), 3)

# TF-IDF Vectorization
df = data.sample(frac=1) # Shuffle data
x = list(df["password"])
vectorizer = TfidfVectorizer(analyzer="char")
X_tfidf = vectorizer.fit_transform(x)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Combine features
df_final = df_tfidf.copy()
df_final["length"] = df["length"].values # Ensure correct alignment after shuffling
df_final["lower_frequency"] = df["lower_frequency"].values
y = df["strength"]
Model Training
A Logistic Regression model with a multinomial classification strategy is used to classify password strength.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(df_final, y, test_size=0.20)

clf = LogisticRegression(multi_class="multinomial", solver='lbfgs', max_iter=1000) # Added solver and max_iter for convergence
clf.fit(X_train, y_train)
Prediction
The trained model can predict the strength of a new password.

def predict_password_strength(password):
  sample_array = np.array([password])
  sample_matrix = vectorizer.transform(sample_array)
  length_pass = len(password)
  length_normalize_lowercase = len([char for char in password if char.islower()]) / len(password)

  # Get other frequency features for the new password
  length_normalize_uppercase = len([char for char in password if char.isupper()]) / len(password)
  length_normalize_number = len([char for char in password if char.isnumeric()]) / len(password)
  special_chars = []
  for char in password:
      if not char.isdigit() and not char.isalpha():
          special_chars.append(char)
  length_normalize_special = len(special_chars) / len(password)

  # Create a DataFrame for the new password with the same column structure as training data
  new_password_features = pd.DataFrame(sample_matrix.toarray(), columns=vectorizer.get_feature_names_out())
  new_password_features["length"] = length_pass
  new_password_features["lower_frequency"] = length_normalize_lowercase
  # Add other engineered features
  new_password_features["upper_frequency"] = length_normalize_uppercase
  new_password_features["number_frequency"] = length_normalize_number
  new_password_features["special_frequency"] = length_normalize_special

  # Predict
  result = clf.predict(new_password_features)

  if result == 0:
    return "Password is weak"
  elif result == 1:
    return "Password is medium"
  else:
    return "Password is strong"

# Example Usage:
# print(predict_password_strength("MyStrongPassword123!"))
Evaluation
The model's performance is evaluated using accuracy, confusion matrix, and a classification report.

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
Usage
To use the password strength predictor:

Clone the repository.

Install dependencies: Ensure you have pandas, numpy, scikit-learn, matplotlib, and seaborn installed.

Run the notebook: Execute the cells in the provided Jupyter/Colab notebook in order.

Use the predict_password_strength function: Call the function with any password string to get its predicted strength.

strength = predict_password_strength("your_password_here")
print(strength)
