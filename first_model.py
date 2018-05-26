import pandas as pd



# path to data
main_file_path = 'train.csv' 

# Reading data
data = pd.read_csv(main_file_path)


# Summary of data  
print(data.describe())

#---------------------------------------


# Selecting and Filtering Data


# List of all columns/features
print(data.columns)


# Select a Single column
data_yearSold = data.YrSold
print(data_yearSold)

# Selecting Multiple columns
columns_of_interest = ['YrSold','SalePrice']
two_columns_of_data = data[columns_of_interest]
# Show shortened result
print(two_columns_of_data.head())

#describing data
two_columns_summary = two_columns_of_data.describe()
print(two_columns_summary)