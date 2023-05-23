import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score


def calculate_adjusted_r_squared(y_pred, y_true, n_features):
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    adj_r_squared = 1 - (rss / (len(y_true) - n_features - 1)) / (tss / (len(y_true) - 1))
    return adj_r_squared

with open('data.csv') as data:
    csv_list = list(csv.reader(data))

x = []
y = []


for row in csv_list[1:]:
    x_values = [float(val) if val != '' else 0 for val in row[1:7]]
    x.append(x_values)
    if row[7] != '':
        y.append(int(row[7]))

X = np.array(x)
Y = np.array(y)

# Adjust the X and Y dimensions for model fitting
X = X[:100]
Y = Y[:100]

model = LinearRegression()
model.fit(X, Y)
y_pred = model.predict(X)

r2 = r2_score(Y, y_pred)
mse = mean_squared_error(Y, y_pred)

# Cross-validation
cv_scores = cross_val_score(model, X, Y, cv=3)

print("MSE:", mse)
print("R^2 score:", r2)
print("Adjusted R^2 score:", calculate_adjusted_r_squared(y_pred, Y, 7))
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
