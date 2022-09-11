import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn'])
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('Estatura-peso_HyM2.csv')

X = data['H_peso'].values
Y = data['H_estat'].values

# calculate mean of x & y using an inbuilt numpy method mean()
mean_x = np.mean(X)
mean_y = np.mean(Y)

m = len(X)

#Modelo de Sklearn
Xsl = X.reshape(m, 1)
reg = LinearRegression()
reg = reg.fit(Xsl,Y)

Y_pred = reg.predict(Xsl)
r2_square = reg.score(Xsl, Y)


# using the formula to calculate m & c
numer = 0
denom = 0
for i in range(m):
  numer += (X[i] - mean_x) * (Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)

max_x = np.max(X) + 10
min_x = np.min(Y)

# calculating line values x and y
x = np.linspace (min_x, max_x, 100)
y = c + m * x

ss_t = 0 #total sum of squares
ss_r = 0 #total sum of square of residuals

for i in range(int(len(x))): # val_count represents the no.of input x values
  y_pred = c + m * X[i]
  ss_t += (Y[i] - mean_y) ** 2
  ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print (f'r2 = {r2} \nr2sl = {r2_square}')

plt.plot(x, y, color='#58b970', label='Regression Line')
plt.plot(Xsl, Y_pred, color = 'k', label = 'Sklearn RL')
plt.scatter(X, Y, c='#ef5423', label='data points')

plt.xlabel('Peso')
plt.ylabel('Estatura')
plt.legend()
plt.show()