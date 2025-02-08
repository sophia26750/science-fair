import streamlit as st 
import numpy as np
from scipy.optimize import minimize

# Define the objective function to maximize (expected return)
def maximize_expected_return(weights):
    return -np.dot(weights, returns)

# Define the budget constraint
def budget_constraint(weights):
    return np.sum(weights) - 1

# # Define the risk constraint
def risk_constraint(weights):
    portfolio_risk = np.dot(np.square(weights), np.square(std_devs))
    return - (portfolio_risk - risk_tolerance ** 2)

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
    # Specify the initial guess for weights
    initial_weights = np.zeros(num_stocks)

    # Specify bounds for weights
    weight_bounds = [[0, 1]] * num_stocks

    # Define the constraints as a list of dictionaries
    constraints = [{'type': 'eq', 'fun': budget_constraint}, 
                {'type': 'ineq', 'fun': risk_constraint}]

    # Execute the optimization
    result = minimize(maximize_expected_return, initial_weights, method='SLSQP', 
                    constraints=constraints, bounds=weight_bounds, 
                    options={'ftol': 1e-6})

    # Print results
    st.write("\nOptimal stock weighting:")
    st.write([round(weight, 5) for weight in result.x])
    st.write("\nMaximum expected return, as a percentage:")
    st.write(round(-result.fun, 5))
except NameError:
    st.markdown(" ")
except ValueError:
    st.markdown(" ")
