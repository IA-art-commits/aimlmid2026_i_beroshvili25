import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Given data points
data = {
    'xi': [-9.56, -6.13, -4.93, -1.27, 1.16, 2.90, 4.90, 7.24],
    'yi': [1.28, 2.04, -1.00, 1.04, -1.80, 1.32, -3.20, -2.20]
}
df = pd.DataFrame(data)

# Calculate Pearson correlation coefficient
# Using numpy for simplicity and accuracy
correlation_coefficient = np.corrcoef(df['xi'], df['yi'])[0, 1]

# Generate scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['xi'], df['yi'], color='blue', label='Data Points')
plt.title('Scatter Plot of xi vs yi')
plt.xlabel('xi')
plt.ylabel('yi')
plt.grid(True)
plt.axhline(0, color='grey', linewidth=0.5) # Add x-axis
plt.axvline(0, color='grey', linewidth=0.5) # Add y-axis
plt.legend()
plt.show()

print(f"Calculated Pearson Correlation Coefficient (r): {correlation_coefficient:.4f}")