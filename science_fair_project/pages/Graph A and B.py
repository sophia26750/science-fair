import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.markdown("## Graph of Optimal Proportions of Stocks A and B Based on Portfolio Risk Tolerance")
# Generate x-coordinates from 9 to 20 with a step size of 0.1
x = np.arange(9, 20, 0.1)

# Calculate y-coordinates
y1 = (800 - np.sqrt(800 ** 2 - 4 * 500 * (400 - x ** 2))) / 1000
y2 = 1 - y1

# Find the coordinates of the intersection point
intersection_x = x[np.argmin(np.abs(y1 - y2))]
intersection_y = y1[np.argmin(np.abs(y1 - y2))]

# Plot the line for Stock A
plt.plot(x, y1, label='Stock A')

# Plot the line for Stock B
plt.plot(x, y2, label='Stock B')

# Plot the intersection point
plt.plot(intersection_x, intersection_y, 'ro')

# Set the labels for the x-axis and y-axis
plt.xlabel('Portfolio Risk Tolerance')
plt.ylabel('Optimal Weight of Stocks')

# Add a legend with the intersection point label
legend_label = f'Intersection: ({intersection_x:.2f}, {intersection_y:.2f})'
plt.legend(labels=['Stock A', 'Stock B', legend_label], loc='best')

# Display the plot
st.pyplot(plt)
# plt.show()