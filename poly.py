import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def adjusted_r2_score(y_true, y_pred, n, p):

    r2 = 1 - ((np.sum((y_true - y_pred) ** 2)) / (np.sum((y_true - np.mean(y_true)) ** 2)))
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

    return adjusted_r2


with open('data.csv') as data:
    csv_list = list(csv.reader(data))

x1 = np.array([])
x2 = np.array([])
x3 = np.array([])
x4 = np.array([])
x5 = np.array([])
x6 = np.array([])
y = np.array([])

for row in csv_list:
    if(row != csv_list[0]):
        x1 = np.append(x1, float(row[1]))
        x2 = np.append(x2, float(row[2]))
        x3 = np.append(x3, float(row[3]))
        x4 = np.append(x4, float(row[4]))
        x5 = np.append(x5, float(row[5]))
        x6 = np.append(x6, float(row[6]))

        if row[7] == '':
            continue
        y = np.append(y, int(row[7]))


X = np.column_stack((x1[0:100], x2[0:100], x3[0:100], x5[0:100]))

# Polynomial regression: y = b0 + b1*x + b2*x^2 + b3*x^3 + ... + bn*x^n
polynomial_regression = PolynomialFeatures(degree=2)
x_polynomial = polynomial_regression.fit_transform(X)

# Fit
linear_regression = LinearRegression()
linear_regression.fit(x_polynomial, y)

# Predict
y_pred = linear_regression.predict(x_polynomial)

# Calculate adjusted R-squared score
n = len(y)
p = x_polynomial.shape[1] - 1
adjusted_r2 = adjusted_r2_score(y, y_pred, n, p)


# Calculate MSE
mse = mean_squared_error(y, y_pred)

# Cross-validation
cv_scores = cross_val_score(linear_regression, x_polynomial, y, cv=4)
cv_average = np.mean(cv_scores)

print("R-squared score: ", r2_score(y, y_pred))
print("Adjusted R-squared score: ", adjusted_r2)
print("MSE:", mse)
print("Cross-validation Scores:", cv_scores)
print("Cross-validation Average Score:", cv_average)

