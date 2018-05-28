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

# Calculating mean absolute error
from sklearn.metrics import mean_absolute_error

predicted_sale_prices = data_model.predict(X)
# in-sample score
avrg_error = mean_absolute_error(y, predicted_sale_prices)
print("The mean absolute error is",avrg_error)


# Validation data (train/make predictions on new data)
from sklearn.model_selection import train_test_split
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
data_model = DecisionTreeRegressor()
# Fit model
data_model.fit(train_X, train_y)

# Get predicited price on validation data
val_predictions = data_model.predict(val_X)
avrg_error = mean_absolute_error(val_y, val_predictions)
print("The mean absolute error for validation test is",avrg_error)
print('\n')

# ********************************************
# ********************************************
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
	model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state = 0)
	model.fit(predictors_train, targ_train)
	preds_val = model.predict(predictors_val)
	mae = mean_absolute_error(targ_val, preds_val)
	return mae

# Finding the most node with the least ammount of errors (cost function)
for max_leaf_nodes in [5, 50, 500 , 5000]:
	my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
	print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes,my_mae))

