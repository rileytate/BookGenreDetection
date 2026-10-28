# Riley Tate and Harrison Niswander
# CS 59300 - Natural Language Processing
# 10-28-26
# NLP Research Project - Logistic Regression Model

#--------------------------------------------------------------------------------------------
#                                 Data Preprocessing
#--------------------------------------------------------------------------------------------
# If throws error when trying to load, use pip install --upgrade kagglehub

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "BooksDatasetClean.csv"

# Load the latest version
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "elvinrustam/books-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

# drop all columns except for description and category
df = df[['Description', 'Category']]

# drop rows in dataset where value is null in description or genre column
df = df.dropna(subset=['Description', 'Category'])

# lowercase description & category data
df = df['Description'].str.lower()
df = df['Category'].str.lower()

# remove extra whitespace
df = df['Description'].str.replace(r'\s+', ' ', regex=True).str.strip()
df = df['Category'].str.replace(r'\s+', ' ', regex=True).str.strip()

# perform TF-IDF Vectorization on data
from sklearn.feature_extraction.text import TfidfVectorizer


#--------------------------------------------------------------------------------------------
#                                 Logistic Regression
#--------------------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# make target class label (y) and remaining features as input features (x) [except for song_title and artist]
y = df['Category']
x = df['Description']

# logistic regression model
model = LogisticRegression(max_iter=1000)

# initialize 10-fold Stratified cross validation
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# perform cross validation
scores = cross_val_score(model, x, y, cv=cv_strategy, scoring="f1")

# results
print("F1 per fold:", scores)
print(f"Mean F1: {np.mean(scores):.4f}")
print(f"Standard Deviation: {np.std(scores):.4f}")