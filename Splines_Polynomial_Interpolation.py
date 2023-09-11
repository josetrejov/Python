import numpy as np
from matplotlib.pyplot import *

# Get user input for the number of data points
num_points = int(input("Enter the number of data points: "))

# Generate random x and y coordinates within the range [0, 360] for the specified number of data points
xk = np.sort(np.random.uniform(0, 360, num_points))
yk = np.sin(xk * np.pi / 180.)  # Calculate corresponding y values as sin(x)
n = len(xk) - 1
hk = np.array([xk[i + 1] - xk[i] for i in range(n)])  # Calculate differences between x values
print(xk, yk, hk)

# Define functions for cubic spline interpolation
def abcd(yk, hk, yppk):
    ak = yk
    bk = np.array([(yk[i + 1] - yk[i]) / hk[i] - (2 * yppk[i] + yppk[i + 1]) * hk[i] / 6. for i in range(len(hk))])
    ck = yppk * 0.5
    dk = np.array([(yppk[i + 1] - yppk[i]) / hk[i] / 6 for i in range(len(hk))])
    return ak, bk, ck, dk

def S(x, a, b, c, d, xk):
    i = 0
    while i < len(xk) - 1 and x >= xk[i]:  # Find the interval that contains x
        i += 1
    i -= 1
    dx = x - xk[i]
    return a[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))  # Evaluate cubic spline using Horner's method

# Calculate Periodic Cubic Spline
A = np.zeros((n, n))  # Initialize matrix A
b = np.zeros(n)  # Initialize vector b

# Build the matrix A and vector b for the linear system
A[0, 0] = 2 * (hk[0] + hk[n - 1])
A[0, 1] = hk[0]
A[0, -1] = hk[n - 1]
b[0] = 6 * ((yk[1] - yk[0]) / hk[0] - (yk[n] - yk[n - 1]) / hk[n - 1])

for i in range(1, n):
    A[i, i] = 2 * (hk[i - 1] + hk[i])
    A[i, i - 1] = hk[i - 1]
    if i < n - 1:
        A[i, i + 1] = hk[i]
    b[i] = 6 * ((yk[i + 1] - yk[i]) / hk[i] - (yk[i] - yk[i - 1]) / hk[i - 1])
    A[-1, 0] = hk[n - 1]

z = np.linalg.solve(A, b)  # Solve the linear system to find second derivatives (y'' values)
yppn = z[0]
ypp = np.append(z, yppn)

print("\nSolution for y''_0 ... y''_n:", ypp)
(a, b, c, d) = abcd(yk, hk, ypp)  # Compute coefficients a, b, c, and d

# Visualize results
xkplot = np.linspace(xk[0], xk[-1], 600)  # Generate points for plotting
spline = [S(x, a, b, c, d, xk) for x in xkplot]  # Compute cubic spline values
ax = subplot(1, 1, 1)
ax.set_title("Cubic spline interpolation")
ax.set_xlabel("x [degrees]")
ax.set_ylabel("y")
ax.grid(linestyle="dotted")

ax.plot(xkplot, spline, label="Spline")
ax.plot(xkplot, np.sin(xkplot * np.pi / 180.), label="sin(x)")
ax.plot(xk, yk, "o", label="knots")
ax.legend(loc="upper right")
show()