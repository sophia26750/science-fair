import streamlit as st 
import numpy as np
from scipy.optimize import minimize

st.write("Lagragian Multipliers for Uncorrelated stocks")

# Input the number of stocks in the portfolio
try: 
    num_stocks = st.text_input("Enter the number of stocks in the portfolio: ", placeholder="Type number here...")
    int_num_stocks = int(num_stocks)
# num_stocks = int(input("Enter the number of stocks in the portfolio: "))
except ValueError:
    st.text_input("Please enter for the next step", placeholder="Type something here...")


try: 
    for i in range(int_num_stocks):  
        returns_input = st.number_input(f"Enter the expected return of {int_num_stocks} Stocks:", key = f"num {i} ")                                     
except NameError:
    st.markdown("#### Please enter number of stocks to continue!")
   
# [float(input(f"Enter the standard deviation of Stock {i + 1}: ")) 
#             for i in range(num_stocks)]

# # Input the portfolio risk tolerance
# risk_tolerance = float(input("Enter the risk tolerance: "))

# # Define the objective function to maximize (expected return)
# def maximize_expected_return(weights):
#     return -np.dot(weights, returns)

# # Define the budget constraint
# def budget_constraint(weights):
#     return np.sum(weights) - 1

# # Define the risk constraint
# def risk_constraint(weights):
#     portfolio_risk = np.dot(np.square(weights), np.square(std_devs))
#     return - (portfolio_risk - risk_tolerance ** 2)

# # Specify the initial guess for weights
# initial_weights = np.zeros(num_stocks)

# # Specify bounds for weights
# weight_bounds = [[0, 1]] * num_stocks

# # Define the constraints as a list of dictionaries
# constraints = [{'type': 'eq', 'fun': budget_constraint}, 
#                {'type': 'ineq', 'fun': risk_constraint}]

# # Execute the optimization
# result = minimize(maximize_expected_return, initial_weights, method='SLSQP', 
#                   constraints=constraints, bounds=weight_bounds, 
#                   options={'ftol': 1e-6})

# # Print results
# print("\nOptimal stock weighting:")
# print([round(weight, 5) for weight in result.x])
# print("\nMaximum expected return, as a percentage:")
# print(round(-result.fun, 5))