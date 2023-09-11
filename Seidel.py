import numpy as np
import matplotlib.pyplot as plt

# Input recommendations
print("Input Recommendations:")
print("1. Ensure that the coefficient matrix is diagonally dominant for convergence.")
print("2. If the system is not diagonally dominant, consider reordering equations.")
print("3. Choose a tolerance level (e.g., 1e-5) to determine convergence.")
print("4. Set a reasonable maximum number of iterations (e.g., 100) to avoid infinite loops.")
print("5. Enter initial guesses for the solution vector if applicable.")
print("6. Verify the correctness of the input matrix and vector dimensions.")
print("7. Ensure that the input matrix is non-singular (determinant is not zero).")

# Get the dimensions of the matrix and vectors as input from the user
m = int(input('Value of m (number of equations): '))  # Number of equations
n = int(input('Value of n (number of unknowns): '))  # Number of unknowns

# Initialize matrices and vectors with zeros
matrix = np.zeros((m, n))  # Coefficient matrix
x = np.zeros(m)           # Solution vector
vector = np.zeros(n)       # Right-hand side vector
comp = np.zeros(m)         # Temporary vector for computation
error = []                    # List to store errors

# Prompt the user to input coefficients and the right-hand side vector
print('Enter the coefficient matrix and the right-hand side vector:')
for r in range(0, m):
    for c in range(0, n):
        matrix[(r), (c)] = float(input("Element a[" + str(r+1) + str(c+1) + "]: "))
    vector[(r)] = float(input('b[' + str(r+1) + ']: '))

# Set tolerance and maximum number of iterations
tol = float(input("Tolerance (e.g., 1e-5): "))
itera = int(input("Maximum number of iterations (e.g., 100): "))

# Initialize lists to store iteration numbers and corresponding errors
iteration_numbers = []
errors = []

# Start the Seidel iterative method
k = 0
while k < itera:
    k = k + 1
    error_sum = 0  # Initialize error sum for this iteration
    for r in range(0, m):
        suma = 0
        for c in range(0, n):
            if c != r:
                suma = suma + matrix[r, c] * x[c]
        new_x = (vector[r] - suma) / matrix[r, r]
        error_sum += abs(new_x - x[r])  # Calculate the error for this variable
        x[r] = new_x  # Update x[r]
    print("Iteration", k, "x:", x)

    # Append iteration number and error to lists for visualization
    iteration_numbers.append(k)
    errors.append(error_sum)

    # Check for convergence (based on the maximum error)
    if error_sum < tol:
        print("Converged to tolerance:", tol)
        break

# Plot the convergence process
plt.figure(figsize=(10, 6))
plt.plot(iteration_numbers, errors, marker='o', linestyle='-', color='b')
plt.xlabel('Iteration Number')
plt.ylabel('Error')
plt.title('Convergence of Seidel Method')
plt.grid(True)
plt.show()