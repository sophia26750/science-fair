# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import minimize

# Define the given inputs


# def optimize_portfolio(correlation):
#     # Define the objective function to maximize (expected return)
#     def expected_return(weights):
#         return -1 * sum(weights[i] * expected_returns[i] for i in range(len(weights)))

#     # Define the budget constraint
#     def budget_constraint(weights):
#         return sum(weights) - 1

#     # Define the risk constraint
#     def risk_constraint(weights):
#         portfolio_variance = sum((weights[i] ** 2) * (standard_deviations[i] ** 2) for i in range(len(weights)))
#         for i in range(len(weights)):
#             for j in range(i + 1, len(weights)):
#                 portfolio_variance += 2 * weights[i] * weights[j] * standard_deviations[i] * standard_deviations[j] * correlation
#         return -(portfolio_variance - risk_tolerance ** 2)

#     # Specify the initial guess and bounds for weights
#     initial_weights = [1/len(expected_returns)] * len(expected_returns)
#     weight_bounds = [(0, 1)] * len(expected_returns)

#     # Define the constraints as a list of dictionaries
#     constraints = [
#         {'type': 'eq', 'fun': budget_constraint},  
#         {'type': 'ineq', 'fun': risk_constraint}
#     ]

#     # Execute the optimization to find the optimal weights
#     result = minimize(expected_return, initial_weights, method='SLSQP', 
#                       constraints=constraints, bounds=weight_bounds,
#                       options={'ftol': 1e-6})

#     return result.x

# # Generate correlation coefficients from -1 to 1 with a step size of 0.1
# x = np.arange(-1, 1.1, 0.1)

# # Calculate optimal weights for the stocks
# weights_optimized = [optimize_portfolio(coeff) for coeff in x]

# # Extract weights for each stock
# weights_stock_1 = [round(weights[0], 5) for weights in weights_optimized]
# weights_stock_2 = [round(weights[1], 5) for weights in weights_optimized]
# weights_stock_3 = [round(weights[2], 5) for weights in weights_optimized]

# # Plotting the results
# plt.xlabel('Correlation Coefficient Between Stocks')
# plt.ylabel('Optimal Weight of Stocks')

# plt.plot(x, weights_stock_1, label="Stock 1")
# plt.plot(x, weights_stock_2, label="Stock 2")
# plt.plot(x, weights_stock_3, label="Stock 3")

# plt.legend()
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Modified function to calculate optimal portfolio
def optimize_portfolio(correlation):
    # Define the objective function to maximize (expected return)
    def expected_return(weights):
        return -1 * sum(weights[i] * returns[i] for i in range(len(weights)))  # Negative to maximize return

    # Define the budget constraint (weights sum to 1)
    def budget_constraint(weights):
        return sum(weights) - 1

    # Define the risk constraint (portfolio volatility under risk tolerance)
    def risk_constraint(weights):
        portfolio_variance = 0
        for i in range(len(weights)):
            for j in range(i, len(weights)):
                cov_ij = std_devs[i] * std_devs[j] * (correlation if i != j else 1)
                portfolio_variance += weights[i] * weights[j] * cov_ij
        portfolio_volatility = np.sqrt(portfolio_variance)
        return -(portfolio_volatility - risk_tolerance)  # Return negative value to enforce inequality

    # Specify initial weights and bounds
    initial_weights = [1 / len(returns)] * len(returns)
    weight_bounds = [(0, 1)] * len(returns)

    # Define the constraints
    constraints = [
        {'type': 'eq', 'fun': budget_constraint},  # Weights must sum to 1
        {'type': 'ineq', 'fun': risk_constraint}   # Risk must be under tolerance
    ]

    # Perform optimization
    result = minimize(expected_return, initial_weights, method='SLSQP',
                      constraints=constraints, bounds=weight_bounds,
                      options={'ftol': 1e-6, 'disp': False})

    return result.x

# Stock returns and volatilities (in percentage)
num_stocks = 3
returns = [5.38, 7.65, 7.29]  # These are the example returns for the three stocks
std_devs = [15, 25, 40]  # Diverse volatilities for a greater effect of correlation
risk_tolerance = 30  # Higher risk tolerance

# Generate correlation coefficients from -1 to 1 with a step size of 0.1
x = np.arange(-1, 1.1, 0.1)

# Calculate optimal weights for each correlation coefficient
weights_optimized = [optimize_portfolio(coeff) for coeff in x]

# Extract weights for each stock
weights_stocks = [[round(weights[i], 5) for weights in weights_optimized] for i in range(num_stocks)]

# Debugging: Print the weights for each correlation coefficient
for coeff, weights in zip(x, weights_optimized):
    print(f"Correlation: {coeff}, Weights: {weights}")

# Plotting the results
plt.figure()
plt.xlabel('Correlation Coefficient Between Stocks')
plt.ylabel('Optimal Weight of Stocks')

plt.title("Graph of Optimal Stock Weights as a Function of the Correlation Coefficient")

# Plot each stock's optimal weight
for i in range(num_stocks):
    plt.plot(x, weights_stocks[i], label=f"Stock {i + 1}")

plt.legend()
# Show the plot
plt.show()
