import numpy as np
from matplotlib.pyplot import *

def lagrange(self, x):
    sum = 0
    for i in range(len(data)):
        xi, yi = data[i]
        
        # Define the L(i) function for Lagrange polynomial.
        def L(i):
            Lvalue = 1
            for j in range(len(data)):
                if i == j:
                    continue
                xj, yj = data[j]
                
                # Calculate the Lagrange basis polynomial L(i) for the given data point.
                Lvalue *= (x - xj) / float(xi - xj)
            return Lvalue
        
        # Calculate the Lagrange interpolating polynomial for the given x-value.
        sum += yi * L(i)
    return sum

# Define the data points for interpolation as a NumPy array.
data = np.array([[1, 1], [3, 9], [5, 25], [9, 81], [12, 144]])
z = []

# Loop through 15 x-values for interpolation.
for i in range(15):
    # Interpolate and round the result to 10 decimal places.
    z.append([i, np.round(lagrange(data, i), 10)])
z = np.array(z)

# Plot the original data and the Lagrange interpolation.
plot(data[:, 0], data[:, 1], linestyle="-", label="Datos")   # Plot data.
plot(z[:, 0], z[:, 1], linestyle="-.", label="Lagrange")    # Plot Lagrange interpolation.
legend(loc="upper left")