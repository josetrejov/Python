import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Define the Cauchy function
def cauchy(x):
    return (1 + x**2)**-1

n = int(input("Enter the value of n: "))

# Generate Chebyshev nodes
x = [np.cos(np.pi*(2*k-1)/(2*n)) for k in range(1, n+1)]
x = 5 * np.array(x)

# Calculate corresponding function values
y = cauchy(x)

# Create a BarycentricInterpolator
f = interpolate.BarycentricInterpolator(x, y)

# Generate points for plotting
xnew = np.linspace(-5, 5, 100)
ynew = f(xnew)

# Plot the interpolation results
plt.plot(x, y, '*', xnew, ynew, '-')
plt.title('Chebyshev Polynomial Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Interpolation Nodes', 'Interpolation Curve'])
plt.show()