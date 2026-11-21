import pandas

# Set the path to the file you'd like to load
file_path = "data.csv"

# Load CSV file
df = pandas.read_csv(file_path)

# drop all columns except for summary and genre
df = df[['summary', 'genre']]

# drop rows in dataset where value is null in summary or genre column
df = df.dropna(subset=['summary', 'genre'])

# Turn the columns into a list
genres = df['genre'].tolist()

# Store all the categories in a list
categories = []

for genre in genres:
    # Check if the label has been recorded
    if genre not in categories:
        categories.append(genre)


# Print the categories
print(categories)
print(len(categories))

with open("output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(categories) + "\n")