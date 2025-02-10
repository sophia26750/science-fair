# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import minimize

# # Define the given inputs
# expected_returns = [12.6, 23.1, 18.4]
# standard_deviations = [23.12, 45.32, 21.6]
# risk_tolerance = 26

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

# User inputs
num_stocks = int(input("Enter the number of stocks in the portfolio: "))
expected_returns = []
standard_deviations = []

for i in range(num_stocks):
    expected_return = float(input(f"Enter the expected return of Stock {i + 1}: "))
    standard_deviation = float(input(f"Enter the standard deviation of Stock {i + 1}: "))
    expected_returns.append(expected_return)
    standard_deviations.append(standard_deviation)

risk_tolerance = float(input("Enter the risk tolerance: "))

