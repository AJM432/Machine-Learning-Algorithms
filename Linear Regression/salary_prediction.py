# csv data from https://www.kaggle.com/karthickveerakumar/salary-data-simple-linear-regression
# Aims to predict salary using linear regression

from linear_regression import model_lin_reg # user defined module
from matplotlib import pyplot as plt
import random
import csv

# read data
with open('Salary_Data.csv', newline='') as f:
    data = list(csv.reader(f))

# convert data to arrays
x = [i:= float(data[x][0]) for x in range(1, len(data))]
y = [i:= float(data[x][1]) for x in range(1, len(data))]

m, b = model_lin_reg(x, y) # calculate regression line
print(f"m={m}, b={b}")

# Prediction at random index in array
rand_index = random.randint(0, len(x))
y_predict = m*x[rand_index]+b
y_actual = y[rand_index]
print(f"x={x[rand_index]}")
print(f"Prediction: {y_predict}")
print(f"Actual: {y_actual}")

# Plot regression line
plt.scatter(x, y)
plt.plot([x[0], x[-1]], [m*x[0]+b, m*x[-1]+b])
plt.show()
