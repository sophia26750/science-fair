import numpy as np
from scipy.optimize import minimize

# Input the number of stocks in the portfolio
num_stocks = int(input("Enter the number of stocks in the portfolio: "))

# Create a list of expected stock returns
returns = [float(input(f"Enter the expected return of Stock {i + 1}: ")) 
           for i in range(num_stocks)]

# Create a list of stock standard deviations
std_devs = [float(input(f"Enter the standard deviation of Stock {i + 1}: ")) 
            for i in range(num_stocks)]

# Input the portfolio risk tolerance (try lowering it for more diversification)
risk_tolerance = float(input("Enter the risk tolerance: "))

# Modify the objective function to encourage diversification
def maximize_expected_return(weights):
    # The negative sign ensures we are maximizing the return
    return -np.dot(weights, returns) + 0.1 * np.sum(np.square(weights - 1/num_stocks)) + 100 * np.sum(np.abs(weights - 0.5))  # Stronger penalty

# Define the budget constraint
def budget_constraint(weights):
    return np.sum(weights) - 1

# Define the risk constraint
def risk_constraint(weights):
    portfolio_risk = np.dot(np.square(weights), np.square(std_devs))
    return - (portfolio_risk - risk_tolerance ** 2)

# Specify the initial guess for weights (balanced portfolio as a starting point)
initial_weights = np.ones(num_stocks) / num_stocks  # Equal initial weights

# Specify bounds for weights
weight_bounds = [[0, 1]] * num_stocks

# Define the constraints as a list of dictionaries
constraints = [{'type': 'eq', 'fun': budget_constraint}, 
               {'type': 'ineq', 'fun': risk_constraint}]

constraints = [
    {'type': 'eq', 'fun': budget_constraint},
    {'type': 'ineq', 'fun': risk_constraint},
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
