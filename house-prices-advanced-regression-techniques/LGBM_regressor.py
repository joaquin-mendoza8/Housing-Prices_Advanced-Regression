import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

# read in data
data_file_path = ("train.csv")
df_train = pd.read_csv("train.csv")

# drop all columns with a lot of NaNs
df_train.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 
                       'Utilities', 'FireplaceQu'], inplace=True)

# make full/half baths a single value
df_train.insert(len(df_train.columns)-1, "Bath", df_train["FullBath"] + 
                0.5*df_train["HalfBath"])
df_train.drop(["FullBath", "HalfBath"], axis=1, inplace=True)

# put columns w/ null values in list
a = df_train.columns[df_train.isnull().any()]

# for each col in a, fill null values w/ mode of that col (fillna())
for i in a:
    df_train[i] = df_train[i].fillna(df_train[i].mode()[0])  

# transform objects in columns to int64
a = df_train.select_dtypes(include = object)

# transform categorical variables to numericals
for i in a:
    label_encoder = preprocessing.LabelEncoder()
    df_train[i] = label_encoder.fit_transform(df_train[i])
    df_train.drop(columns = [], inplace = True)

# --------------- PREPROCESSING --------------------------------------------

# read test data
test = pd.read_csv("test.csv")

# drop columns with many NaNs
test.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 
                   'Utilities', 'FireplaceQu'], inplace = True)

# make full/half baths a single value
test.insert(len(test.columns)-1, "Bath", test["FullBath"] + 
                0.5*test["HalfBath"])
test.drop(["FullBath", "HalfBath"], axis=1, inplace=True)

# fill rest of NaNs w/ zeros
a = test.columns[test.isnull().any()]
for i in a:
    test[i] = test[i].fillna(test[i].mode()[0])

# change categorical data to numerical
a = test.select_dtypes(include = object)
for i in a:
    label_encoder = preprocessing.LabelEncoder()
    test[i] = label_encoder.fit_transform(test[i])
    test.drop(columns = [], inplace = True) # drop columns that become empty

# ----- MODEL ------------------------------------------------------


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('SalePrice', axis=1), df_train['SalePrice'], test_size=0.2, random_state=1337)

lgbm = LGBMRegressor(objective = 'regression', 
                       num_leaves = 13,
                       learning_rate = 0.034428, 
                       n_estimators = 4235,
                       random_state = 1337)

# fit'n'show rmse

lgbm.fit(X_train, y_train)
lgbm_train_predict = lgbm.predict(X_train)
mse = mean_squared_error(y_train, lgbm_train_predict)
rmse = np.sqrt(mse)
lgbm_test_predict = lgbm.predict(X_test)
r2 = r2_score(y_test, lgbm_test_predict)


print("About to write to /output")

# open the output file for writing
with open('/output/lgbm_output.txt', 'w') as f:
    f.write("Root mean squared error: {}\n".format(rmse))
    f.write("R^2 Score: {}\n".format(r2))

# write to console to confirm file creation and writing
print("Successfully created and wrote to /output/lgbm_output.txt")