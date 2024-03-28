import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns
from joblib import dump

# Load the car crashes dataset
car_crashes = sns.load_dataset('car_crashes')

# Remove the specified features: 'no_previous', 'ins_premium', 'ins_losses'
features = car_crashes.drop(['total', 'abbrev', 'no_previous', 'ins_premium', 'ins_losses'], axis=1)
target = car_crashes['total']  # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(max_depth=5)
dt_regressor.fit(X_train, y_train)

# Initialize and train the Linear Regression model
lr_regressor = LinearRegression()
lr_regressor.fit(X_train, y_train)

# Save your models to disk
dump(dt_regressor, 'dt_regressor.joblib')
dump(lr_regressor, 'lr_regressor.joblib')
