# Riley Tate and Harrison Niswander
# CS 59300 - Natural Language Processing
# 10-28-26
# NLP Research Project - Transformer Model

#--------------------------------------------------------------------------------------------
#                                 Data Preprocessing
#--------------------------------------------------------------------------------------------
# If throws error when trying to load, use pip install --upgrade kagglehub
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas

# Set the path to the file you'd like to load
file_path = "BooksDatasetClean.csv"

# Load CSV file
df = pandas.read_csv(file_path)

# drop all columns except for description and category
df = df[['Description', 'Category']]

# drop rows in dataset where value is null in description or genre column
df = df.dropna(subset=['Description', 'Category'])

# Check if the given label is a correct option
def compare_labels(y_actual, y_pred):
    # y_actual should be a comma-separated string containing multiple labels
    # y_pred should be a single label

    # Start by separating the labels into multiple labels
    actual_labels = y_actual.split(' , ')

    # Go through each of the actual labels and see if the predicted label matches
    for label in actual_labels:
        if(label == y_pred):
            # If our predicted label is one of the books label, count it as a success
            return True

    # If our label does not match, return false
    return False