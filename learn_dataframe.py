import pandas as pd

# Sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# Selecting the first row
first_row = df.iloc[0]

# Selecting multiple rows (first three)
first_three_rows = df.iloc[0:3]

# Selecting specific rows and columns (first three rows of column 'B' and 'C')
specific_selection = df.iloc[0:3, [1, 2]]

# Selecting the last row
last_row = df.iloc[-1]

print(first_row)
print()
print(first_three_rows)
print()
print(specific_selection)
print()
print(last_row)
print()
