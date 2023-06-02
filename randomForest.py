import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


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

rf = RandomForestRegressor(random_state=57)
rf.fit(X, y)

# Perform prediction on the training data
predictions = rf.predict(X)

scores = cross_val_score(rf, X, y, cv=8)
r2_score = rf.score(X, y)
adjusted_r2_score = calculate_adjusted_r2(r2_score, X.shape[0], X.shape[1])

# Calculate mean squared error
mse = mean_squared_error(y, predictions)

print("R2 Score:", r2_score)
print("Adjusted R2 Score:", adjusted_r2_score)
print("Mean Squared Error:", mse)
print("Cross-Validation Scores:", scores)
print("Average Cross-Validation Score:", np.mean(scores))


