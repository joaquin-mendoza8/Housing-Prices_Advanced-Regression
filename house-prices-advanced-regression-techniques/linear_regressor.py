import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import data_preprocessing

df = data_preprocessing.df_train

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('SalePrice', axis=1), df['SalePrice'], test_size=0.2, random_state=1337)

# create linear regression model
linear = LinearRegression()

# train the model on the training data
linear.fit(X_train, y_train)

# make predictions on the testing data
y_pred = linear.predict(X_test)

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("Root mean squared error:", rmse)
print("Mean squared error:", mse)
print("R^2 score:", r2)