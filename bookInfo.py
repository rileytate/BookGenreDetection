import pandas

# Set the path to the file you'd like to load
file_path = "BooksDatasetClean.csv"

# Load CSV file
df = pandas.read_csv(file_path)

# drop all columns except for description and category
df = df[['Description', 'Category']]

# drop rows in dataset where value is null in description or genre column
df = df.dropna(subset=['Description', 'Category'])

# Turn the columns into a list
book_categories = df['Category'].tolist()

# Store all the categories in a list
categories = []

for book in book_categories:
    # Split the string
    labels = book.split(' , ')

    # Go through each label
    for label in labels:
        label = label.strip()

        # Check if the label has been recorded
        if label not in categories:
            categories.append(label)


# Print the categories
print(categories)
print(len(categories))

with open("output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(categories) + "\n")