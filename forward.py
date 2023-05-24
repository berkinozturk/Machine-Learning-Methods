import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# read the data from CSV file
data = pd.read_csv('data.csv')

# drop rows with NaN values in any column
data = data.dropna()

# extract the independent and dependent variables
X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y = data['Y']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

# apply standardization to the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# perform forward stepwise selection
n_features = X_train_std.shape[1]
selected = []
for i in range(n_features):
    remaining = [x for x in range(n_features) if x not in selected]
    best_score = 0
    for candidate in remaining:
        model = LinearRegression()
        features = selected + [candidate]
        X_temp = X_train_std[:, features]
        model.fit(X_temp, y_train)
        score = model.score(X_temp, y_train)
        if score > best_score:
            best_score = score
            best_feature = candidate
    selected.append(best_feature)

# select the best features and fit the model
X_train_selected = X_train_std[:, selected]
X_test_selected = X_test_std[:, selected]
model = LinearRegression()
model.fit(X_train_selected, y_train)


y_pred = model.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
mse_normalized = mse / (np.max(y) - np.min(y))**2
r2 = r2_score(y_test, y_pred)


feature_names = [f"x{i+1}" for i in selected]
print("Selected features (best to worst):", feature_names)
print(f"R2 Score on Test Set: {r2:.2f}")
print("MSE: {:.2f}".format(mse_normalized))




