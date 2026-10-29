# Riley Tate and Harrison Niswander
# CS 59300 - Natural Language Processing
# 10-28-26
# NLP Research Project - Decision Tree Model

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
df['Description'] = df['Description'].str.lower()
df['Category'] = df['Category'].str.lower()

# remove extra whitespace
df['Description'] = df['Description'].str.replace(r'\s+', ' ', regex=True).str.strip()
df['Category'] = df['Category'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Keep the 20 most common categories with enough examples for 10-fold CV
counts = df["Category"].value_counts()
categories = counts[counts >= 10].nlargest(20).index
df = df[df["Category"].isin(categories)].copy()

#--------------------------------------------------------------------------------------------
#                                 Decision Tree
#--------------------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# perform TF-IDF Vectorization on data
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=50_000, min_df=2)
x_vectorized = vectorizer.fit_transform(df['Description'])

# encode the target string data into numeric labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Category'])

# decision tree model
model = DecisionTreeClassifier(random_state=42)

# initialize 10-fold Stratified cross validation
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# perform cross validation
scores = cross_val_score(model, x_vectorized, y_encoded, cv=cv_strategy, scoring="f1_macro", n_jobs=1)

# results
print("F1 per fold:", scores)
print(f"Mean F1: {np.mean(scores):.4f}")
print(f"Standard Deviation: {np.std(scores):.4f}")