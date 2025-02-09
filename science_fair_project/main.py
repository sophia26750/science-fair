import streamlit as st 
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt



# Function to calculate portfolio risk for a given set of weights
def calculate_portfolio_risk(weights, std_devs):
    # Portfolio risk formula: sqrt(w1^2 * sigma1^2 + w2^2 * sigma2^2 + ...)
    return np.sqrt(np.sum(np.square(weights) * np.square(std_devs)))

# Define the objective function to maximize (expected return)
def maximize_expected_return(weights):
    diversification_penalty = 0.05 * np.sum(np.square(weights - 1/num_stocks))  
    return -np.dot(weights, returns) + diversification_penalty  # Maximize return

# Define the budget constraint
def budget_constraint(weights):
    return np.sum(weights) - 1

# # Define the risk constraint
def risk_constraint(weights):
    portfolio_risk = np.dot(np.square(weights), np.square(std_devs))
    return - (portfolio_risk - risk_tolerance ** 2)

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


st.markdown("# Lagrangian Multipliers for Uncorrelated stocks")
st.markdown("---")

# Input the number of stocks in the portfolio
try: 
    str_num_stocks = st.text_input("Enter the number of stocks in the portfolio (integer please): ", placeholder="Type number here...")
    num_stocks = int(str_num_stocks)
# num_stocks = int(input("Enter the number of stocks in the portfolio: "))
except ValueError:
    st.text(" ")

try: 

    returns = [st.number_input(f"Enter the expected return of {i + 1} Stocks as percent:", key = f"num {i} ", placeholder="0.00")
                                for i in range(num_stocks)]  
    # returns = float(str_returns)                                   
except NameError:
    st.markdown("#### Please enter number of stocks to continue!")
except ValueError:
    st.markdown(" ")

try:
    
    std_devs = [st.number_input(f"Enter the standard deviation of Stock {i + 1}: ", key = f"num {i + num_stocks} ", placeholder="0.00")
                    for i in range(num_stocks)]
    # std_devs = float(str_std_devs)
except NameError:
    st.markdown(" ")
except ValueError:
    st.markdown(" ")

st.markdown("---")

# Input the portfolio risk tolerance
try:
    st.write("Select the risk tolerance: ")
    risk_tolerance = st.slider("You can use arrows if needed", 0, 100, 50)
    st.write(f"Chosen risk tolerance:  {risk_tolerance}")
    # str_risk_tolerance = st.text_input("Enter the risk tolerance: ")
    # risk_tolerance = float(str_risk_tolerance)
except NameError:
    st.markdown(" ")
except ValueError:
    st.markdown(" ")



try: 
    # Specify the initial guess for weights (balanced portfolio as a starting point)
    initial_weights = np.ones(num_stocks) / num_stocks  # Equal initial weights

    # Specify bounds for weights
    weight_bounds = [[0, 1]] * num_stocks

    # Define the constraints as a list of dictionaries
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
    st.write("\nOptimal stock weighting:")
    optimized_weights = ([round(weight, 5) for weight in result.x])
    st.write(optimized_weights)
    st.write("\nMaximum expected return, as a percentage:")
    st.write(round(-result.fun, 5))
except NameError:
    st.markdown(" ")
except ValueError:
    st.markdown(" ")

st.markdown("Graph of Portfolio Risk as a Function of Proportion of Stock A")

try:

    # Calculate the risk of the optimized portfolio
    optimized_risk = calculate_portfolio_risk(optimized_weights, std_devs)

    # Set the x-coordinates from 0 to 1 with a step size of 0.01
    x = np.arange(0, 1, 0.01)

    # Calculate corresponding portfolio risk for each combination of weights
    y = np.zeros_like(x)
    for i, weight_A in enumerate(x):
        # Calculate remaining weights for other stocks
        weights = np.array([weight_A] + [(1 - weight_A) / (num_stocks - 1)] * (num_stocks - 1))  # Ensure sum of weights = 1
        y[i] = calculate_portfolio_risk(weights, std_devs)

    # Find the lowest point on the graph
    lowest_point_idx = np.argmin(y)
    lowest_point_x = x[lowest_point_idx]
    lowest_point_y = y[lowest_point_idx]

    # Set labels for the plot
    plt.xlabel('Weight of Stock A')
    plt.ylabel('Portfolio Risk (%)')

    # Plot the points for the risk curve
    plt.plot(x, y, label="Risk Curve")

    # Mark the lowest point on the graph (min risk)
    plt.plot(lowest_point_x, lowest_point_y, 'ro', label=f'Lowest Risk: ({lowest_point_x:.2f}, {lowest_point_y:.2f})')

    # Annotate the coordinates of the lowest point
    plt.annotate(f'({lowest_point_x:.2f}, {lowest_point_y:.2f})', (lowest_point_x, lowest_point_y),
                textcoords="offset points", xytext=(4, 10), ha='center')

    # Plot the optimized portfolio result (using the calculated risk)
    plt.plot(np.sum(optimized_weights), optimized_risk, 'bo', label=f'Optimized Portfolio: ({np.sum(optimized_weights):.2f}, {optimized_risk:.2f})')

    # Add labels and legend
    plt.legend()
    plt.title("Risk Curve with Optimized Portfolio")

    # Show the plot
    st.pyplot(plt)

except NameError:
    st.markdown("Finish Part One")
except ValueError:
    st.markdown("Finish Part One")