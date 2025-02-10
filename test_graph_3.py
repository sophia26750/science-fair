import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Given inputs
expected_returns = [12.6, 23.1, 18.4]
standard_deviations = [23.12, 45.32, 21.6]
risk_tolerance = 26

def maximize_return(correlation_matrix):
    # Define the objective function to maximize (expected return)
    def expected_return(weights):
        return -1 * sum([w * r for w, r in zip(weights, expected_returns)])

    # Define the budget constraint (weights sum to 1)
    def budget_constraint(weights):
        return sum(weights) - 1

    # Define the risk constraint
    def risk_constraint(weights):
        variance = sum([
            w1 * w2 * correlation_matrix[i][j] * standard_deviations[i] * standard_deviations[j]
            for i, w1 in enumerate(weights) for j, w2 in enumerate(weights)
        ])
        return -(variance - risk_tolerance**2)

    # Initial guess and bounds for weights
    initial_weights = [1/3] * len(expected_returns)
    weight_bounds = [(0, 1)] * len(expected_returns)

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': budget_constraint},
        {'type': 'ineq', 'fun': risk_constraint}
    ]

    # Optimization
    result = minimize(expected_return, initial_weights, method='SLSQP',
                      constraints=constraints, bounds=weight_bounds,
                      options={'ftol': 1e-6})

    return -1 * result.fun

# Generate correlation matrices and calculate the maximum expected return for each
correlation_values = np.round(np.arange(-1, 1.1, 0.1), 1)
max_returns = []

for corr in correlation_values:
    correlation_matrix = np.array([
        [1, corr, corr],
        [corr, 1, corr],
        [corr, corr, 1]
    ])
    max_return = maximize_return(correlation_matrix)
    max_returns.append(round(max_return, 5))

# Plotting the results
plt.xlabel('Correlation Coefficient Between Stocks')
plt.ylabel('Maximum Expected Portfolio Return (%)')

plt.plot(correlation_values, max_returns)
plt.show()
