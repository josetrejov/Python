from matplotlib.pyplot import *
import numpy as np
import sympy as sym

# Generate random data with a linear trend.
x = np.linspace(0, 1, 20)
y = 3 + 5 * x + np.random.standard_normal([1, 20])

# Initialize variables for summation.
sumy, sumxy, sumx, sumxx = 0, 0, 0, 0

# Calculate sums for later use in linear regression.
for i in range(len(y[0])):
    sumy += y[0, i]
    sumx += x[i]
    sumxy += x[i] * y[0, i]
    sumxx += x[i] * x[i]

# Create a matrix for linear regression calculations.
mat = np.array([[np.shape(y)[1], sumx, sumy], [sumx, sumxx, sumxy]])

# Function to perform Gaussian elimination on a matrix.
def operaciones(matriz, shape):
    for i in range(len(matriz) - 1):
        for j in range(len(matriz) - 1):
            j = j + 1
            if (j + i) > shape[0] - 1:
                break
            else:
                if matriz[j + i, i] == 0:
                    a = np.copy(matriz[j + i])
                    if (j + i) < shape[0] - 1:
                        matriz[j + i] = matriz[j + i + 1]
                        matriz[j + i + 1] = a
                    pivot = matriz[j + i, i] / matriz[i, i]
                    matriz[j + i] = matriz[j + i] - pivot * matriz[i]
                elif matriz[i, i] == 0:
                    continue
                else:
                    pivot = matriz[j + i, i] / matriz[i, i]
                    matriz[j + i] = matriz[j + i] - pivot * matriz[i]
    return matriz

# Initialize a matrix for partial pivoting.
mat_p = np.zeros((4, 4))

# Function to perform Gaussian elimination on a matrix and return the result.
def gauss(matriz):
    shape = np.shape(matriz)
    Final_mat = operaciones(matriz, shape)
    return Final_mat[0]

# Perform Gaussian elimination on the matrix 'mat' to solve for the coefficients of the linear equation.
print(gauss(mat))

# Calculate the slope (m) and y-intercept (b) for the linear equation.
m = round(mat[1, 2] / mat[1, 1], 4)
b = round((mat[0, 2] - (m * mat[0, 1])) / mat[0, 0], 4)

# Create a formula string representing the linear equation.
formula = ('y = ', m, 'x +', b)

# Calculate the linear regression line.
f = m * x + b

# Plot the original data points and the linear regression line.
plot(x, y[0], '*', label='Data Points')
plot(x, f, linestyle="-", label=formula)
legend(loc="upper left")
title('Linear Regression')
