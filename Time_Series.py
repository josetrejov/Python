import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Import data
df = pd.read_table('TS1.csv', encoding='latin-1', delimiter=',')
df.head()

# Set the date as the table index and establish the frequency
df.index = pd.to_datetime(df['Period'], format='%d.%m.%Y')
df.index.freq = pd.infer_freq(df.index)
df.head()

# Convert all "Null" values to "np.nan"
df = df.replace('Null', np.nan)

# Data type conversion
df[['Revenue', 'Sales_quantity', 'Average_cost', 'The_average_annual_payroll_of_the_region']] = df[['Revenue', 'Sales_quantity', 'Average_cost', 'The_average_annual_payroll_of_the_region']].astype(float)
df.dtypes

# Handle missing values using forward-fill
df.fillna(method='ffill', inplace=True)

# Visualize the data
plt.plot(df.index, df['Revenue'], label='Revenue')
plt.plot(df.index, df['Sales_quantity'], label='Sales')
plt.plot(df.index, df['Average_cost'], label='Average_cost')  # Changed label to 'Average_cost'
plt.plot(df.index, df['The_average_annual_payroll_of_the_region'], label='Payroll')
plt.legend()
plt.title('"ABC-123" Company')
plt.grid()

"""
Temporal decomposition involves extracting the following components:
*   **Level**: The average value of the time series
*   **Trend**: Whether the value in the series is increasing or decreasing
*   **Seasonal**: Short-term repetitive patterns in the series
*   **Residual**: The random variation in the series
"""

# Perform seasonal decomposition for each column
for column in ['Revenue', 'Sales_quantity', 'Average_cost', 'The_average_annual_payroll_of_the_region']:
    series = df[column]
    result = seasonal_decompose(series, model="additive")
    result.plot()
    plt.title(f'Decomposition of {column}')
    plt.show()