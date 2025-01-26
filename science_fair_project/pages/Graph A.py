import streamlit as st  
import matplotlib.pyplot as plt
import numpy as np

st.markdown("## Graph of Portfolio Risk as a Function of Proportion of Stock A")

# Set the x-coordinates from 0 to 1 with a step size of 0.01
x = np.arange(0, 1, 0.01)

# Calculate the corresponding y-coordinates
y = np.sqrt(100 * x ** 2 + 400 * (1 - x) ** 2)

# Find the lowest point on the graph
lowest_point_idx = np.argmin(y)
lowest_point_x = x[lowest_point_idx]
lowest_point_y = y[lowest_point_idx]

# Set labels for the plot
plt.xlabel('Weight of Stock A')
plt.ylabel('Portfolio Risk (%)')

# Plot the points
plt.plot(x, y)

# Mark the lowest point on the graph
plt.plot(lowest_point_x, lowest_point_y, 'ro')

# Annotate the coordinates of the lowest point
plt.annotate(f'({lowest_point_x:.2f}, {lowest_point_y:.2f})', (lowest_point_x, lowest_point_y),
             textcoords="offset points", xytext=(4, 10), ha='center')


# Display the plot
st.pyplot(plt)
# plt.show()
