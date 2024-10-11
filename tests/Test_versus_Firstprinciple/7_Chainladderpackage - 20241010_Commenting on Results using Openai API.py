import os
import pandas as pd
import numpy as np
import chainladder as cl
from datetime import datetime
import openai
import os, sys
from openai import OpenAI

# Configuration
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your actual API key
os.environ["OPENAI_MODEL_NAME"] = "gpt-4"

client = OpenAI(api_key="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")

openai.api_key = os.getenv("OPENAI_API_KEY")
INPUT_FILE_PATH = "sample_motor_claims_data.csv"  # Ensure this file is in the same directory as the script
valuation_date = datetime(2023, 12, 31)

def get_gpt_analysis(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a senior actuarial manager and data analyst providing insights on insurance data analysis."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def csv_to_dataframe(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    if os.stat(file_path).st_size == 0:
        raise ValueError(f"The file {file_path} is empty.")
    
    try:
        df = pd.read_csv(file_path, delimiter=',')
        
        if df.empty:
            raise ValueError("The CSV file contains no data.")
        
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

def save_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved output to {filename}")

# Data Loading and Preprocessing
claims_df = csv_to_dataframe(INPUT_FILE_PATH)

# Convert date columns to datetime
date_columns = ['incurred_date', 'reported_date', 'paid_date']
for col in date_columns:
    if claims_df[col].dtype == 'object':
        claims_df[col] = pd.to_datetime(claims_df[col], format='%Y-%m-%d', errors='coerce')

# Create origin and development columns
claims_df['origin'] = claims_df['incurred_date'].dt.to_period('Y').astype(str)
claims_df['development'] = claims_df['paid_date'].where(claims_df['paid_date'] <= valuation_date, valuation_date)
claims_df['development'] = claims_df['development'].dt.to_period('Y').astype(str)

# Data Quality Report
data_quality_prompt = f"""
Analyze the following data quality aspects of the claims dataset:
1. Missing values: {claims_df.isnull().sum()}
2. Data types: {claims_df.dtypes}
3. Date range: Incurred dates from {claims_df['incurred_date'].min()} to {claims_df['incurred_date'].max()}
4. Claim count by year: {claims_df['origin'].value_counts().sort_index()}

Provide a comprehensive data quality report, highlighting potential issues and their impact on the analysis.
"""

data_quality_report = get_gpt_analysis(data_quality_prompt)
print("Data Quality Report:")
print(data_quality_report)
save_to_file(data_quality_report, "data_quality_report.txt")

# Create the triangle
runoff_triangle = cl.Triangle(claims_df, 
                        origin='origin', 
                        development='development', 
                        columns=['paid_amount'], cumulative=False)

triangle = runoff_triangle.incr_to_cum()

# Development Factors Analysis
dev = cl.Development().fit(triangle)
ldf = dev.ldf_

development_factors_prompt = f"""
Analyze the following development factors:
{ldf}

Provide insights on:
1. The overall trend of the development factors
2. Any unusual patterns or outliers
3. Potential external factors that could have influenced these development factors
4. Implications for the reserving process
"""

development_factors_analysis = get_gpt_analysis(development_factors_prompt)
print("\nDevelopment Factors Analysis:")
print(development_factors_analysis)
save_to_file(development_factors_analysis, "development_factors_analysis.txt")

# Chain Ladder Method
BCL_method = cl.Chainladder().fit(triangle)

bcl_results_prompt = f"""
Analyze the results of the Basic Chain Ladder method:
1. Ultimate claims: {BCL_method.ultimate_}
2. IBNR: {BCL_method.ibnr_}

Provide insights on:
1. The adequacy of the estimated reserves
2. Any patterns or trends in the ultimate claims by accident year
3. Potential limitations of the Basic Chain Ladder method for this dataset
4. Recommendations for further analysis or alternative methods
"""

bcl_results_analysis = get_gpt_analysis(bcl_results_prompt)
print("\nBasic Chain Ladder Results Analysis:")
print(bcl_results_analysis)
save_to_file(bcl_results_analysis, "bcl_results_analysis.txt")

# Final Report
final_report_prompt = f"""
Compile a comprehensive actuarial report based on the following analyses:
1. Data Quality Report
2. Development Factors Analysis
3. Basic Chain Ladder Results Analysis

The report should include:
1. Executive summary
2. Key findings and insights
3. Recommendations for further analysis or action items
4. Potential risks and limitations of the current analysis
"""

final_report = get_gpt_analysis(final_report_prompt)
print("\nFinal Actuarial Report:")
print(final_report)
save_to_file(final_report, "final_actuarial_report.txt")

print("\nAnalysis complete. All reports have been saved as text files in the current directory.")