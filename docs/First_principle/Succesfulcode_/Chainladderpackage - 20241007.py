######################################################################################################
import os


#https://www.youtube.com/watch?v=q6QLGS306d0&ab_channel=TylerAI Crew ai tutorial that the code is based on
os.environ["OPENAI_API_KEY"] = "api key here"
os.environ["OPENAI_MODEL_NAME"]= "gpt-4o"

# Actuarial Analysis Script
# This script processes motor claims data, performs data quality checks,
# creates run-off triangles, and calculates ultimate claims and reserves.

# Import required libraries
import pandas as pd
import numpy as np
import xlwings as xw
import chainladder as cl
import openai
from datetime import datetime

# Configuration
# -------------
# Set your OpenAI API key here
openai.api_key = 'api-key-here'

# Set the path to the input CSV file
INPUT_FILE_PATH = r"C:\Users\masho\OneDrive\AI\Deloitte_framework\2_Analysis\Attempt5_AI_python_freeestyle\Freestyle\sample_motor_claims_data.csv"
valuation_date = datetime(2023, 12, 31)
# Data Loading
# ------------

def csv_to_dataframe(file_path):
    """
    Read a comma-delimited CSV file and convert it to a pandas DataFrame.
    
    Args:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: A pandas DataFrame containing the claims data.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Check if the file is empty
    if os.stat(file_path).st_size == 0:
        raise ValueError(f"The file {file_path} is empty.")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, delimiter=',')
        
        # Basic data validation
        if df.empty:
            raise ValueError("The CSV file contains no data.")
        
        # Display basic information about the DataFrame
        print(f"Successfully loaded data from {file_path}")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("Column names:")
        print(df.columns.tolist())
        
        return df
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {file_path} is empty or contains no valid data.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing the CSV file: {str(e)}")

claims_df = csv_to_dataframe(INPUT_FILE_PATH)

#print(claims_df)
#print(claims_df.dtypes)
# Convert date columns to datetime
date_columns = ['incurred_date', 'reported_date', 'paid_date']
for col in date_columns:
    if claims_df[col].dtype == 'object':
        claims_df[col] = pd.to_datetime(claims_df[col], format='%Y-%m-%d', errors='coerce')



# Create origin and development columns
claims_df['origin'] = claims_df['incurred_date'].dt.to_period('Y').astype(str)
claims_df['development'] = claims_df['paid_date'].where(claims_df['paid_date'] <= valuation_date, valuation_date)
claims_df['development'] = claims_df['development'].dt.to_period('Y').astype(str)

print(claims_df.dtypes)
print(claims_df)

# Create the triangle
runoff_triangle = cl.Triangle(claims_df, 
                        origin='origin', 
                        development='development', 
                        columns=['paid_amount'], cumulative=False)

print(runoff_triangle.is_cumulative) #confirm that the triangle is incremental
triangle = runoff_triangle.incr_to_cum() #convert incremental to cumulative

print(triangle)


triangle.link_ratio.incr_to_cum() #convert link ratio to cumulative

##############################
#Development factors
dev = cl.Development().fit(triangle)
dev.ldf_

# Example of next steps:
BCL_method = cl.Chainladder().fit(triangle) #https://chainladder-python.readthedocs.io/en/latest/getting_started/tutorials/deterministic-tutorial.html

BCL_method.ultimate_
BCL_method.ibnr_

print(BCL_method.ultimate_)
print(BCL_method.ibnr_)
