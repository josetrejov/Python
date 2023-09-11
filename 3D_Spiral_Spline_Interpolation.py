from numpy import arange, cos, linspace, pi, sin, random
from scipy.interpolate import splprep, splev
import pylab

# Generate data for an ascending spiral in 3D space
t = linspace(0, 1.75 * 2 * pi, 100)  # Create a time parameter
x = sin(t)  # X-coordinate of the spiral
y = cos(t)  # Y-coordinate of the spiral
z = t       # Z-coordinate of the spiral

# Add random noise to the data
x += random.normal(scale=0.1, size=x.shape)
y += random.normal(scale=0.1, size=y.shape)
z += random.normal(scale=0.1, size=z.shape)

# Define spline interpolation parameters
s = 3.0   # Smoothness parameter
k = 2     # Spline order
nest = -1 # Estimate of the number of knots needed (-1 = maximal)

# Find the knot points and create the spline representation
tckp, u = splprep([x, y, z], s=s, k=k, nest=nest)

# Evaluate the spline to obtain interpolated points
xnew, ynew, znew = splev(linspace(0, 1, 400), tckp)

# Create subplots and plot the data and spline interpolation
pylab.subplot(2, 2, 1)
data, = pylab.plot(x, y, 'bo-', label='Original Data')
fit, = pylab.plot(xnew, ynew, 'r-', label='Spline Fit')
pylab.legend()
pylab.xlabel('X')
pylab.ylabel('Y')

pylab.subplot(2, 2, 2)
data, = pylab.plot(x, z, 'bo-', label='Original Data')
fit, = pylab.plot(xnew, znew, 'r-', label='Spline Fit')
pylab.legend()
pylab.xlabel('X')
pylab.ylabel('Z')

pylab.subplot(2, 2, 3)
data, = pylab.plot(y, z, 'bo-', label='Original Data')
fit, = pylab.plot(ynew, znew, 'r-', label='Spline Fit')
pylab.legend()
pylab.xlabel('Y')
pylab.ylabel('Z')

pylab.show()