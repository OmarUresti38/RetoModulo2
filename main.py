import numpy as np
import math
import operator
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('test.csv')

X = data['Age'].values
Y = data['SibSp'].values

# calculate mean of x & y using an inbuilt numpy method mean()
mean_x = np.mean(X)
mean_y = np.mean(Y)

m = len(X)

# using the formula to calculate m & c
numer = 0
denom = 0
for i in range(m):
  numer += (X[i] - mean_x) * (Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)

print (f'm = {m} \nc = {c}')

max_x = np.max(X) + 10
min_x = np.min(Y)

# calculating line values x and y
x = np.linspace (min_x, max_x, 76)
y = c + m * x

plt.plot(x, y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='#ef5423', label='data points')

plt.xlabel('Age')
plt.ylabel('Siblings')
plt.legend()
plt.show()

ss_t = 0 #total sum of squares
ss_r = 0 #total sum of square of residuals

for i in range(int(len(x))): # val_count represents the no.of input x values
  y_pred = c + m * X[i]
  ss_t += (Y[i] - mean_y) ** 2
  ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print(r2)