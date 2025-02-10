import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Function to calculate portfolio risk for a given set of weights
def calculate_portfolio_risk(weights, std_devs):
    return np.sqrt(np.sum(np.square(weights) * np.square(std_devs)))

# Modify the objective function to maximize expected return
def maximize_expected_return(weights):
    diversification_penalty = 0.05 * np.sum(np.square(weights - 1/num_stocks))  # Weakened but still encouraging diversification
    return -np.dot(weights, returns) + diversification_penalty  # Maximize return

# Define the budget constraint
def budget_constraint(weights):
    return np.sum(weights) - 1  # Weights must sum to 1

# Define the risk constraint
def risk_constraint(weights):
    portfolio_risk = np.dot(np.square(weights), np.square(std_devs))
    return - (portfolio_risk - risk_tolerance ** 2)  # Risk tolerance constraint

# Define the non-negative return constraint (portfolio return must be >= 0)
def non_negative_return_constraint(weights):
    portfolio_return = np.dot(weights, returns)
    return portfolio_return  # Must be >= 0 (returns >= 0)

# Add a constraint to prevent any stock from getting too much weight
def max_weight_constraint(weights):
    return np.max(weights) - 0.5  # No stock can have more than 50% of the portfolio

# Add a constraint for minimum allocation per stock (e.g., 5%)
def min_weight_constraint(weights):
    return np.min(weights) - 0.05  # No stock can have less than 5%

# Input the number of stocks in the portfolio and their properties
num_stocks = 4  # Update based on user input
returns = [23, 12, 43, 4]  # Update based on user input
std_devs = [12, 3, 43, 23]  # Update based on user input
risk_tolerance = 34  # Portfolio risk tolerance, update based on user input

# Specify the initial guess for weights (balanced portfolio as a starting point)
initial_weights = np.ones(num_stocks) / num_stocks  # Equal initial weights

# Specify bounds for weights (each weight must be between 0 and 1)
weight_bounds = [(0, 1)] * num_stocks

# Define the constraints
constraints = [
    {'type': 'eq', 'fun': budget_constraint},  # Budget constraint
    {'type': 'ineq', 'fun': risk_constraint},  # Risk tolerance constraint
    {'type': 'ineq', 'fun': non_negative_return_constraint},  # Non-negative return constraint
    {'type': 'ineq', 'fun': max_weight_constraint},  # Max weight constraint to prevent too much concentration
    {'type': 'ineq', 'fun': min_weight_constraint}  # Minimum weight constraint to force allocation to all stocks
]

# Execute the optimization
result = minimize(maximize_expected_return, initial_weights, method='SLSQP', 
                  constraints=constraints, bounds=weight_bounds, 
                  options={'ftol': 1e-6, 'maxiter': 10000})  # Increased max iterations

# Get optimized weights for each stock
optimized_weights = [round(weight, 5) for weight in result.x]

# Calculate the portfolio risk based on the optimized weights
optimized_risk = calculate_portfolio_risk(optimized_weights, std_devs)

# Print the results
print("\nOptimal stock weighting:")
print(optimized_weights)
print("\nMaximum expected return, as a percentage:")
print(round(-result.fun, 5))
print("\nOptimized portfolio risk:", optimized_risk)

# Now we will graph the optimal weights and risk for each stock

# Generate x-coordinates (risk tolerance values)
x = np.arange(9, 40, 0.1)  # Adjust the risk tolerance range as needed

# Calculate y-coordinates for the optimal weight of each stock (just for illustration)
y = np.zeros((num_stocks, len(x)))

# Calculate the optimal weights for each risk tolerance value
for i, risk in enumerate(x):
    # Update the risk constraint for each value of x (risk tolerance)
    constraints[1] = {'type': 'ineq', 'fun': lambda weights: -calculate_portfolio_risk(weights, std_devs) + risk}

    # Re-optimize for this risk tolerance
    result = minimize(maximize_expected_return, initial_weights, method='SLSQP', 
                      constraints=constraints, bounds=weight_bounds, 
                      options={'ftol': 1e-6, 'maxiter': 10000})

    # Store the optimized weights for each stock
    for j in range(num_stocks):
        y[j, i] = result.x[j]

# Plot the results for each stock
for j in range(num_stocks):
    plt.plot(x, y[j], label=f'Stock {j+1}')

# Add legend for intersection
intersection_x = x[np.argmin(np.abs(y[0] - y[1]))]
intersection_y = y[0, np.argmin(np.abs(y[0] - y[1]))]
legend_label = f'Intersection: ({intersection_x:.2f}, {intersection_y:.2f})'
plt.legend(loc='best')

# Adjust the y-axis range
plt.ylim(0, 3)  # Adjust the range as needed

# Add labels and legend
plt.xlabel('Portfolio Risk Tolerance')
plt.ylabel('Optimal Weight of Stocks')
plt.legend(loc='best')
plt.title("Optimal Stock Weights Based on Risk Tolerance")

# Display the plot
plt.show()
