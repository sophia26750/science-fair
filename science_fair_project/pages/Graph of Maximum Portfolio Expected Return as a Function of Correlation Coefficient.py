import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import streamlit as st 

st.markdown("## Graph of Maximum Portfolio Expected Return as a Function of Correlation Coefficient")

def maximize_return(correlation):
    # Define the objective function to maximize (expected return)
    def expected_return(weights):
        weight_a, weight_b = weights
        return -1 * (20 * weight_a + 30 * weight_b)    
    
    # Define the budget constraint
    def budget_constraint(weights):
        weight_a, weight_b = weights
        return weight_a + weight_b - 1

    # Define the risk constraint
    def risk_constraint(weights):
        weight_a, weight_b = weights
        return -(100 * weight_a ** 2 + 400 * weight_b ** 2 + \
                 2 * weight_a * weight_b * 10 * 20 * correlation - 225)
 
    # Specify the initial guess and bounds for weights
    initial_weights = [0, 0]
    weight_bounds = [[0, 1], [0, 1]]

    # Define the constraints as a list of dictionaries
    constraints = [
        {'type': 'eq', 'fun': budget_constraint},  
        {'type': 'ineq', 'fun': risk_constraint}
    ]

    # Execute the optimization to find the optimal weight for Stock A
    result = minimize(expected_return, initial_weights, method='SLSQP', 
                      constraints=constraints, bounds=weight_bounds,
                      options={'ftol': 1e-6})
  
    return -1 * result.fun

# Generate correlation coefficients from -1 to 1 with a step size of 0.1
x = np.round(np.arange(-1, 1.1, 0.1), 1)

# Calculate the corresponding maximum expected return
y = [round(maximize_return(coeff), 5) for coeff in x]

# Plotting the results
plt.xlabel('Correlation Coefficient Between Stocks')
plt.ylabel('Maximum Expected Portfolio Return')

plt.plot(x, y)
# plt.show()
st.pyplot(plt)