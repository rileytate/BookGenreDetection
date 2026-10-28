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
df_cleaned = df.dropna(subset=['Description', 'Category'])
