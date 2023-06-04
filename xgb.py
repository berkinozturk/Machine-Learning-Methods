from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import csv
from sklearn.model_selection import train_test_split, cross_val_score


def calculate_adjusted_r2(r2_score, num_samples, num_features):
    adjusted_r2 = 1 - ((1 - r2_score) * (num_samples - 1)) / (num_samples - num_features - 1)
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
    if row != csv_list[0]:
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
y = y[0:100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=60)

# train model
model = XGBRegressor()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)


r2 = model.score(X_test, y_test) # Calculate R2 score
print('R^2 Score:', r2)

num_samples = X_test.shape[0]
num_features = X_test.shape[1]
adjusted_r2 = calculate_adjusted_r2(r2, num_samples, num_features) # Calculate adjusted R2 score
print('Adjusted R^2 Score:', adjusted_r2)

mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
print("Mean Squared Error (MSE):", mse)

cv_scores = cross_val_score(model, X, y, cv=10)  # Calculate CV scores
cv_average = np.mean(cv_scores)  # Calculate CV average
print("Cross-Validation (CV) Average:", cv_average)

X_final = np.column_stack((x1[100:120], x2[100:120], x3[100:120], x5[100:120]))
y_pred_final = model.predict(X_final)
print("Predictions: ", y_pred_final)