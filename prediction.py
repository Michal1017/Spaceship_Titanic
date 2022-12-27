import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# read train and test data from csv files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print informations about dataset
print(train.head())

print("Shape of train dataset:")
print(train.shape)

print("Shape of test dataset:")
print(test.shape)

print(train.info())

# get features and output information from train dataframe
y = train['Transported'].astype('int').to_frame()
X = train.drop(['Transported', 'PassengerId', 'Name', 'Cabin'], axis=1)


# get total money spent on spacespip for each passenger
X['total_spent'] = X['RoomService'] + X['FoodCourt'] + \
    X['ShoppingMall'] + X['Spa'] + X['VRDeck']

# get features from train dataframe
X_test = test.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
X_test['total_spent'] = X_test['RoomService'] + X_test['FoodCourt'] + \
    X_test['ShoppingMall'] + X_test['Spa'] + X_test['VRDeck']


print(X.head())
print(y.head())

# get numerical features
num_cols = [col_name for col_name in X.columns if X[col_name].dtype in [
    'int64', 'float64']]

print("Columns with numerical values:")
print(num_cols)


# get categorical features
cat_cols = [col_name for col_name in X.columns if X[col_name].dtype in ['object']]

print("Columns with categorical values:")
print(cat_cols)

print('Number of unique values in columns with categorical values:')
print(X[cat_cols].nunique())

print("Number of missing data in train dataset:")
print(X.isnull().sum())

print("Number of missing values in y:")
print(y.isnull().sum())

print("Number of missing data in test dataset:")
print(X_test.isnull().sum())

# create dataframe with y and X
all_train = pd.concat([X, y], sort=False, axis=1)

print(all_train)

# get correlation matrix for all features
print(pd.DataFrame(
    abs(all_train.corr()['Transported']).sort_values(ascending=False)))

# split datafame on train and validation dataset
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2)

# create dataframe with y and X
all_train = pd.concat([X_train, y_train], sort=False, axis=1)

print(all_train)

# get sorted correlation of fuetures with y information
print(pd.DataFrame(
    abs(all_train.corr()['Transported']).sort_values(ascending=False)))

# exchange empty numerical information with mean of specific feature
numerical_transform = SimpleImputer(strategy='mean')

# pipeline for categorical features first exchange nulls with most frequent value then use one hot encoder for machine learning model
categorical_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# merge all data together
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transform, num_cols),
    ('cat', categorical_transform, cat_cols)
])

# hyperparameters
learning_rate = 0.1
n_estimators = 500

# create model
model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                      use_label_encoder=False)

# create pipeline for model training
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# train model
xgb_model.fit(X_train, y_train)

# check prediction on validation dataset
y_valid_pred_xgb = xgb_model.predict(X_valid)

print("Accuracy score for validation dataset for XGBClassifer model: ",
      accuracy_score(y_valid, y_valid_pred_xgb))

# predict y for test dataframe
y_pred = xgb_model.predict(X_test)

print(y_pred)

# share output information to csv file
submission = pd.DataFrame(
    {'Transported': y_pred.astype(bool)}, index=test['PassengerId'])

print(submission)

submission.to_csv('submission.csv')
