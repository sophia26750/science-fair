import numpy as np
from scipy.optimize import minimize


# Modify the objective function to encourage diversification and balance
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

# Input the number of stocks in the portfolio
num_stocks = int(input("Enter the number of stocks in the portfolio: "))

# Create a list of expected stock returns
returns = [float(input(f"Enter the expected return of Stock {i + 1}: ")) 
           for i in range(num_stocks)]

# Create a list of stock standard deviations
std_devs = [float(input(f"Enter the standard deviation of Stock {i + 1}: ")) 
            for i in range(num_stocks)]

# Input the portfolio risk tolerance
risk_tolerance = float(input("Enter the risk tolerance: "))



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

# Execute the optimization with more iterations and higher precision
result = minimize(maximize_expected_return, initial_weights, method='SLSQP', 
                  constraints=constraints, bounds=weight_bounds, 
                  options={'ftol': 1e-6, 'maxiter': 10000})  # Increased max iterations

# Print results
print("\nOptimal stock weighting:")
print([round(weight, 5) for weight in result.x])
print("\nMaximum expected return, as a percentage:")
print(round(-result.fun, 5))

