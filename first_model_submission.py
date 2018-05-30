import numpy as pd
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the training data
train = pd.read_csv('train.csv')

#pull data into target (y) and predictors (X)
train_y = train.SalePrice

# Features we are using to predict the Sale Price of a house
predictors_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']



# Create training predictors data
train_X = train[predictors_cols]

my_model = RandomForestRegressor()

# Fit the model: Capture patterns from provided data. This is the heart of modeling.
my_model.fit(train_X, train_y)

# Read test data
test = pd.read_csv('test.csv')

# Pull the same features/columns as training data from the test data
test_X = test[predictors_cols]

# Use the model to make predictions
predicted_prices = my_model.predict(test_X)

print(predicted_prices)

#************************
#		Submission
#************************

# Submissions usally have two columns, ID and the prediction column
# ID comes from test data. Prediction column will use target_field?

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# print(my_submission)

# save into a file called submission
my_submission.to_csv('submission.csv', index=False)
print('Finished!')