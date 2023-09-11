import numpy as np
import matplotlib.pyplot as plt

# Define data points
datosx = np.array([0.15, 2.3, 3.15, 4.85, 6.25, 7.95], dtype=float)
datosy = np.array([4.79867, 4.49013, 4.2243, 3.47313, 2.66674, 1.51909], dtype=float)

# Create a copy of datosy for Neville interpolation
datosy2 = datosy.copy()

# Function to calculate divided differences for interpolation
def calculate_differences(x_values, y_values):
    n = len(x_values)
    differences = np.zeros((n, n))

    # Initialize the first column with y_values
    differences[:, 0] = y_values

    # Calculate divided differences
    for j in range(1, n):
        for i in range(n - j):
            differences[i][j] = (differences[i + 1][j - 1] - differences[i][j - 1]) / (x_values[i + j] - x_values[i])

    return differences[0]

# Calculate divided differences for Newton interpolation
diff_values = calculate_differences(datosx, datosy)

# Input from the user for interpolation
print('NEWTON AND NEVILLE INTERPOLATION ALGORITHMS.')
print('============================================')
print()
print('Enter the value for interpolation: ')
interpolar = eval(input())

# Newton Interpolation Function
def newton_interpolation(x_values, y_values, interp_value):
    n = len(x_values)
    result = y_values[0]
    temp = 1

    for i in range(1, n):
        temp *= (interp_value - x_values[i - 1])
        result += temp * diff_values[i]

    return result

# Calculate the result using Newton interpolation
valor = newton_interpolation(datosx, datosy, interpolar)

# Print the result of Newton interpolation
print(f'For {interpolar}, we obtain {valor} with Newton interpolation. ')

# Neville Interpolation Function
def neville_interpolation(x_values, y_values, interp_value):
    n = len(x_values)
    p = np.zeros(n)

    for i in range(n):
        p[i] = y_values[i]

    for k in range(1, n):
        for i in range(n - k):
            p[i] = ((interp_value - x_values[i + k]) * p[i] + (x_values[i] - interp_value) * p[i + 1]) / (x_values[i] - x_values[i + k])

    return p[0]

# Calculate the result using Neville interpolation
valorN = neville_interpolation(datosx, datosy2, interpolar)

# Print the result of Neville interpolation
print(f'For {interpolar}, we obtain {valorN} with Neville interpolation. ')

# Calculate x and y limits for plotting
x_min = min(np.min(datosx), interpolar) - 1
x_max = max(np.max(datosx), interpolar) + 1
y_min = min(np.min(datosy), valor, valorN) - 1
y_max = max(np.max(datosy), valor, valorN) + 1

# Disable LaTeX text rendering in Matplotlib
plt.rcParams['text.usetex'] = False

# Create a figure for the plot
plt.figure(figsize=(10, 6))

# Plot the original data points within the specified x and y limits
plt.scatter(datosx, datosy, label='Data Points', color='blue', marker='o')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Create an array of x values for the interpolation curve within the specified x limits
x_interp = np.linspace(x_min, x_max, 100)

# Compute the Newton interpolation values for the given x_interp
y_newton = [newton_interpolation(datosx, datosy, xi) for xi in x_interp]

# Compute the Neville interpolation values for the given x_interp
y_neville = [neville_interpolation(datosx, datosy2, xi) for xi in x_interp]

# Plot the Newton interpolation curve
plt.plot(x_interp, y_newton, label='Newton Interpolation', color='green')

# Plot the Neville interpolation curve
plt.plot(x_interp, y_neville, label='Neville Interpolation', color='red')

# Add labels and title to the plot
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolation Comparison')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()