import pandas as pd



# path to data
main_file_path = 'train.csv' 

# Reading data
data = pd.read_csv(main_file_path)


# Summary of data  
print(data.describe())

# *********************************
# ********************************


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

# ********************************************
# ********************************************

# Building model #

# Prediction target/ column we want to predict
y = data.SalePrice

# loading predictors
#fireplace, fullbath, yearbuilt

data_predictors = ['YearBuilt','FullBath','Fireplaces']

X = data[data_predictors]


# Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# Fit: Capture patterns from provided data. This is the heart of modeling.
# Predict: Just what it sounds like
# Evaluate: Determine how accurate the model's predictions are.

from sklearn.tree import DecisionTreeRegressor 

# Define model
data_model = DecisionTreeRegressor()

# Fit model
data_model.fit(X,y)

# parameters aboyt the type of model built
# print(data_model.fit(X,y))
print('\n')
print("Making predictions for the first 5 houses")
print(X.head(),'\n')
print("The predictions are")
print(data_model.predict(X.head()))

