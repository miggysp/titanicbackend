import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the car crashes dataset
car_crashes = sns.load_dataset('car_crashes')

# Let's use all the numeric features available for prediction
features = car_crashes.drop(['total', 'abbrev'], axis=1) # Drop non-numeric and target column
target = car_crashes['total'] # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize and train a Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(max_depth=5)
dt_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt_regressor.predict(X_test)

# Evaluate the Decision Tree Regressor
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f'Decision Tree Regressor MSE: {mse_dt:.2f}')
print(f'Decision Tree Regressor R^2: {r2_dt:.2f}')

# Initialize and train a Linear Regression model
lr_regressor = LinearRegression()
lr_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred_lr = lr_regressor.predict(X_test)

# Evaluate the Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f'Linear Regression MSE: {mse_lr:.2f}')
print(f'Linear Regression R^2: {r2_lr:.2f}')
