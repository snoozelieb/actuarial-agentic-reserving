import os
import pandas as pd
import numpy as np
import chainladder as cl
from datetime import datetime
import openai
import os, sys
from openai import OpenAI

api = "api key here"
# Configuration
os.environ["OPENAI_API_KEY"] = api  # Replace with your actual API key
os.environ["OPENAI_MODEL_NAME"] = "gpt-4"

client = OpenAI(api_key=api)

openai.api_key = os.getenv(api)
INPUT_FILE_PATH = r"C:\Users\masho\OneDrive\AI\Deloitte_framework\2_Analysis\Attempt5_AI_python_freeestyle\Freestyle\sample_motor_claims_data.csv"

valuation_date = datetime(2023, 12, 31)
############################################################################################################
# Actuarial Reserving using Swarm and Chainladder
# This script performs various actuarial reserving exercises using the chainladder package and the Swarm framework.

# Import required libraries
import os
import pandas as pd
import chainladder as cl
from swarm import Swarm, Agent
from openai import OpenAI
from datetime import datetime

# Initialize OpenAI client
# Make sure to set your OPENAI_API_KEY as an environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Swarm
swarm = Swarm()

# Function Definitions

def import_data(context_variables):
    """
    Imports data from a CSV file and converts it to a chainladder Triangle object.
    
    :param context_variables: Dictionary containing the file path
    :return: chainladder Triangle object or error message
    """
    file_path = context_variables.get("file_path")
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, delimiter=',')

        date_columns = ['incurred_date', 'reported_date', 'paid_date']
        for col in date_columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')



        # Create origin and development columns
        df['origin'] = df['incurred_date'].dt.to_period('Y').astype(str)
        df['development'] = df['paid_date'].where(df['paid_date'] <= valuation_date, valuation_date)
        df['development'] = df['development'].dt.to_period('Y').astype(str)
        # Convert to chainladder Triangle
        triangle = cl.Triangle(df, origin='origin', development='development', columns='paid_amount', cumulative=False)
        return triangle
    except Exception as e:
        return f"Error importing data: {str(e)}"

def basic_chainladder(context_variables):
    """
    Applies the basic chainladder method to the triangle.
    
    :param context_variables: Dictionary containing the triangle
    :return: Full triangle after applying chainladder or error message
    """
    triangle = context_variables.get("triangle")
    try:
        # Apply Development factor method
        model = cl.Development().fit(triangle)
        return model.full_triangle
    except Exception as e:
        return f"Error in basic chainladder: {str(e)}"

def bornhuetter_ferguson(context_variables):
    """
    Applies the Bornhuetter-Ferguson method to the triangle.
    
    :param context_variables: Dictionary containing the triangle
    :return: Ultimate loss estimates or error message
    """
    triangle = context_variables.get("triangle")
    try:
        # Apply Bornhuetter-Ferguson method
        model = cl.BornhuetterFerguson().fit(triangle)
        return model.ultimate_
    except Exception as e:
        return f"Error in Bornhuetter-Ferguson: {str(e)}"

def bootstrapping(context_variables):
    """
    Performs bootstrapping analysis on the triangle.
    
    :param context_variables: Dictionary containing the triangle
    :return: Bootstrapped ultimate loss estimates or error message
    """
    triangle = context_variables.get("triangle")
    try:
        # Perform bootstrapping
        model = cl.BootstrapODPSample(n_sims=1000, random_state=42).fit(triangle)
        return model.ultimate_
    except Exception as e:
        return f"Error in bootstrapping: {str(e)}"

def apply_selection(context_variables):
    """
    Applies selection criteria to remove certain link ratios.
    
    :param context_variables: Dictionary containing the triangle and link ratio threshold
    :return: Selected triangle or error message
    """
    triangle = context_variables.get("triangle")
    link_ratio_threshold = context_variables.get("link_ratio_threshold", 2.0)
    try:
        # Apply selection criteria
        selected_triangle = triangle.link_ratio[(triangle.link_ratio < link_ratio_threshold) & (triangle.link_ratio > 0)]
        return selected_triangle
    except Exception as e:
        return f"Error in applying selection: {str(e)}"

# Define the Actuarial Agent
data_agent = Agent(
    name="ActuarialAgent",
    instructions="You are an actuarial agent who takes in claims data and performs reserving exercising using the Basic chain ladder method.",
    functions=[import_data, basic_chainladder, bornhuetter_ferguson, bootstrapping, apply_selection],)

# Chain Ladder Specialist Agent
chainladder_agent = Agent(
    name="ChainLadderSpecialist",
    instructions="You are a specialist in the Chain Ladder method. Your expertise lies in applying and interpreting the results of the Chain Ladder technique for loss reserving.",
    functions=[basic_chainladder, apply_selection],
)

# Bornhuetter-Ferguson Specialist Agent
bf_agent = Agent(
    name="BFSpecialist",
    instructions="You are a specialist in the Bornhuetter-Ferguson (BF) method. Your expertise is in applying the BF technique and interpreting its results for loss reserving, especially when dealing with immature or volatile data.",
    functions=[bornhuetter_ferguson],
)

# Bootstrapping Specialist Agent
bootstrap_agent = Agent(
    name="BootstrappingSpecialist",
    instructions="You are a specialist in Bootstrapping techniques for actuarial analysis. Your expertise is in applying bootstrapping methods to generate confidence intervals and assess variability in loss reserve estimates.",
    functions=[bootstrapping],
)

# Selection Specialist Agent
selection_agent = Agent(
    name="SelectionSpecialist",
    instructions="You are a specialist in selecting and applying appropriate criteria for actuarial analysis. Your expertise lies in identifying and implementing selection criteria to refine and improve reserve estimates.",
    functions=[apply_selection],
)

# Main Execution

# Get user input
file_path = r"C:\Users\masho\OneDrive\AI\Deloitte_framework\2_Analysis\Attempt5_AI_python_freeestyle\Freestyle\sample_motor_claims_data.csv"
link_ratio_threshold = float(3)

# Set up context variables
context_variables = {
    "file_path": file_path,
    "link_ratio_threshold": link_ratio_threshold
}

# Import data
print("\nImporting data...")
response = swarm.run(
    messages=[{"role": "user", "content": "Import the claims data and return ONLY the claims triangle in dataframe format"}],
    agent=data_agent,
    context_variables=context_variables,
)
triangle = response.messages[-1]["content"]

print(response)

print(triangle)



context_variable2 = {
     "triangle": triangle
}

############################################################################################################

print("\nTesting data...")
response2 = swarm.run(
    messages=[{"role": "user", "content": "Import the claims data, convert it to the claims triangle in dataframe format and then perform the chain ladder analysis on it"}],
    agent=data_agent,
    context_variables=context_variables,
)
triangle2 = response.messages[-1]["content"]

print(triangle2)

# Chain Ladder analysis
print("\nPerforming Chain Ladder analysis...")
cl_response = swarm.run(
    messages=[{"role": "user", "content": "Perform Chain Ladder analysis on the context triangle and interpret the results."}],
    agent=chainladder_agent,
    context_variables=context_variable2,
)
print("Chain Ladder Results:")
print(cl_response.messages[-1]["content"])

# Bornhuetter-Ferguson analysis
print("\nPerforming Bornhuetter-Ferguson analysis...")
bf_response = swarm.run(
    messages=[{"role": "user", "content": "Perform Bornhuetter-Ferguson analysis and interpret the results."}],
    agent=bf_agent,
    context_variables=context_variable2,
)
print("Bornhuetter-Ferguson Results:")
print(bf_response.messages[-1]["content"])  

# Bootstrapping analysis
print("\nPerforming Bootstrapping analysis...")
bootstrap_response = swarm.run(
    messages=[{"role": "user", "content": "Perform Bootstrapping analysis and interpret the results."}],
    agent=bootstrap_agent,
    context_variables=context_variable2,
)
print("Bootstrapping Results:")
print(bootstrap_response.messages[-1]["content"])

# Selection analysis
print("\nApplying selection criteria...")
selection_response = swarm.run(
    messages=[{"role": "user", "content": f"Apply selection criteria with threshold {link_ratio_threshold} and interpret the results."}],
    agent=selection_agent,
    context_variables=context_variables,
)
print("Selection Results:")
print(selection_response.messages[-1]["content"])

print("\nActuarial analysis complete.")


############################################################################################################

# Usage Instructions:
# 1. Ensure you have installed all required packages: swarm, openai, pandas, and chainladder.
# 2. Set your OpenAI API key as an environment variable named OPENAI_API_KEY.
# 3. Prepare your claims data in a CSV file with columns for 'origin', 'development', and 'incurred'.
# 4. Run this script and provide the requested inputs when prompted.
# 5. The script will perform the actuarial analyses and display the results.