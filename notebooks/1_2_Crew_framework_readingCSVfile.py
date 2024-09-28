"""
Title: Agentic framework used in Claims reserving

Overview
This code is designed to automate the process of actuarial analysis for insurance data. Actuarial analysis involves evaluating financial risks, especially in insurance, by analyzing data related to claims and reserves. The code uses various tools and AI-powered agents to handle different parts of this analysis, making the process more efficient and accurate.

Key Components and Their Roles
Imports and Libraries:

Data Handling & Analysis: Libraries like pandas, numpy, and scipy help in managing and analyzing data.
Visualization: matplotlib and seaborn are used to create graphs and charts.
Actuarial Methods: chainladder and reservespy provide specialized functions for actuarial calculations.
Machine Learning & Preprocessing: sklearn helps in preparing data for analysis.
AI and Automation: pyautogen and openai are used to create and manage AI-powered agents that assist in the analysis.

AI-Powered Agents:

The code sets up several AI agents, each with a specific role in the actuarial analysis process. These agents are like specialized helpers that perform tasks such as data analysis, calculating development factors, reviewing methods, adjusting risks, compiling reports, and ensuring regulatory compliance.
Example Agents:
DataAnalyst: Analyzes and processes the data.
DevelopmentFactorSpecialist: Calculates factors used to estimate reserves.
ReportCompiler: Compiles all findings into a comprehensive report.
RegulatoryComplianceOfficer: Ensures that all processes meet regulatory standards.

Functions for Data Processing and Analysis:

Data Validation: Checks if the data has the necessary columns and suggests mappings if some are missing.
Data Conversion: Transforms raw claims data into structured formats (like loss triangles) suitable for actuarial methods.
Actuarial Methods Application: Applies techniques like the Chain Ladder and Benktander methods to estimate insurance reserves.
Visualization: Creates heatmaps and other visual representations of the data to help understand trends and patterns.
Risk Adjustment: Calculates adjustments to account for uncertainties in the reserve estimates.

Main Workflow (actuarial_analysis_workflow):

Initialization: Sets up a group chat with all the AI agents to coordinate the analysis.
Data Ingestion: Loads the input data (e.g., from a CSV file) and preprocesses it.
Line of Business Identification: Identifies different segments or categories within the data, such as different types of insurance products.
Triangle Analysis: For each line of business, it creates loss triangles (a common actuarial tool) and analyzes development factors.
Reserving Methods Application: Applies actuarial methods to estimate the reserves needed for future claims.
Risk Adjustment Calculation: Adjusts the reserve estimates to account for risk.
Report Compilation: Gathers all the analysis results into a detailed report.
Regulatory Compliance Review: Ensures that the final report meets all necessary regulatory standards.
Final AI Analysis (Optional): Provides a synthesized overview and insights based on the entire analysis process.
What Can This Code Do?
Automate Data Analysis: Automatically processes and analyzes insurance claims data without manual intervention.
Apply Advanced Actuarial Methods: Uses sophisticated techniques to estimate the financial reserves required for future claims.
Ensure Accuracy and Compliance: Reviews each step to ensure that the analysis is accurate and complies with industry regulations.
Generate Comprehensive Reports: Compiles all findings into detailed reports that can be used for decision-making.
Utilize AI for Enhanced Insights: Leverages AI to assist in complex tasks, provide suggestions, and ensure thorough analysis.

How It Works Together

Data Input: You start by providing an insurance claims data file (e.g., input_data.csv).
Preprocessing: The DataAnalyst agent cleans and organizes the data, identifying key columns and structures.
Analysis: Specialized agents perform various analyses, such as calculating development factors and applying reserving methods.
Review: Other agents review the work to catch any errors or suggest improvements.
Risk Adjustment: Adjustments are made to account for uncertainties, ensuring the estimates are robust.
Reporting: All results are compiled into a report that meets regulatory standards.
Final Review: The RegulatoryComplianceOfficer ensures everything is in order before the report is finalized.

Benefits of This Approach

Efficiency: Automates repetitive and complex tasks, saving time.
Accuracy: Reduces the likelihood of human error through consistent processing.
Compliance: Ensures all analyses meet necessary regulatory requirements.
Scalability: Can handle large datasets and multiple lines of business simultaneously.
Insightful Reporting: Provides comprehensive reports that offer valuable insights for strategic decisions.
Conclusion
In simple terms, this code sets up an automated, AI-driven system to handle the entire process of actuarial analysis for insurance data. It ensures that data is properly processed, analyzed using industry-standard methods, reviewed for accuracy, and compiled into detailed, compliant reports. This automation not only enhances efficiency and accuracy but also provides valuable insights to support informed decision-making in the insurance industry.
"""

#######################################################################################################
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

#https://www.youtube.com/watch?v=q6QLGS306d0&ab_channel=TylerAI Crew ai tutorial that the code is based on
os.environ["OPENAI_API_KEY"] = ("your key")
os.environ["OPENAI_MODEL_NAME"]= "gpt-4o"
#os.environ["SERPER_API_KEY"] = "Your Key" # serper.dev API key
# It can be a local model through Ollama / LM Studio or a remote
# model like OpenAI, Mistral, Antrophic or others (https://docs.crewai.com/how-to/LLM-Connections/)

########################################################################################################
# Define your agents with roles and goals
# FOr each agent, you need a role, goal and a backstory


#######################################################################################################################################
#Reading in CSV file
# Function to read CSV file
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_json(orient='split')
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"

# Get CSV file path from user
csv_file_path = input("Enter the path to your actuarial data CSV file: ")

# Read CSV file
csv_data = read_csv_file(csv_file_path)

#######################################################################################################################################
# Define agents
data_analyst = Agent(
    role='Data Analyst',
    goal='Process and analyze actuarial data for multiple Lines of Business.You should identify data types, assess data quality, provide insights, and guide the user through the process of analyzing and splitting data by LOB.Ensure compliance with APN 401 Section 3 (Data) and consider user inputs for data-driven decisions',
    backstory="""You are a skilled data analyst specializing in actuarial data. 
    You excel at identifying data types, assessing data quality, and providing insights.""",
    verbose=True,
    allow_delegation=False,
    # You can pass an optional llm attribute specifying what model you wanna use.
    # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
)

development_factor_specialist = Agent(
    role='Development Factor Specialist',
    goal='Review the accuracy of the run off triangles created based on input data. Calculate development factors and apply appropriate reserving methods',
    backstory="""You are an experienced actuary specializing in calculating development factors for reserving. 
    You are proficient in methods like Chain Ladder and Bornhuetter-Ferguson.""",
    verbose=True,
    allow_delegation=False,
    
)

risk_adjustment_specialist = Agent(
    role='Risk Adjustment Specialist',
    goal='Calculate risk adjustments using bootstrapping techniques',
    backstory="""You are an expert in calculating risk adjustments for insurance reserves. 
    You specialize in using bootstrapping techniques to calculate adjustments at the 75th percentile.""",
    verbose=True,
    allow_delegation=False,
    
)

report_compiler = Agent(
    role='Report Compiler',
    goal='Compile a comprehensive, regulation-compliant actuarial report',
    backstory="""You are a meticulous report compiler with extensive knowledge of actuarial standards. 
    You excel at integrating complex analyses into clear, compliant reports.""",
    verbose=True,
    allow_delegation=True,
    
)
#######################################################################################################################################
# Define tasks
##Tasks are the actual work that needs to be done by the agents. So we give the agents tasks
#  Parameters for tasks = the description, expected output (How is the information going to presented...Report, dataframe, informale email...) and the agent that will be doing the task
#Crewai allows one to assign a task to a single agent or multiple agents
# Define tasks
task1 = Task(
    description=f"""Analyze the provided actuarial data CSV file. 
    The data is provided in JSON format as follows: {csv_data}
    Identify Lines of Business, assess data quality, and prepare the data for further analysis. 
    Provide a summary of your findings.""",
    agent=data_analyst,
    expected_output="Detailed data analysis report"
)

task2 = Task(
    description="""Using the prepared data from the previous task, calculate development factors and apply appropriate reserving methods. 
    Consider methods like Chain Ladder and Bornhuetter-Ferguson. 
    Provide reserve estimates at best estimate (50th percentile), prudent (75th percentile), and optimistic (25th percentile).""",
    agent=development_factor_specialist,
    expected_output="Reserving analysis report with development factors and reserve estimates"
)

task3 = Task(
    description="""Calculate risk adjustments using bootstrapping techniques at the 75th percentile. 
    Ensure alignment with IFRS 17 and APN 401 Section 6. 
    Document your methods, assumptions, and justification for margins.""",
    agent=risk_adjustment_specialist,
    expected_output="Risk adjustment analysis report"
)

task4 = Task(
    description="""Compile a comprehensive actuarial report integrating all previous analyses. 
    Include sections on data quality, methodologies, reserve calculations, and risk adjustments. 
    Ensure compliance with APN 401 and IFRS 17 standards.""",
    agent=report_compiler,
    expected_output="Final comprehensive actuarial report"
)

# Assemble the crew
crew = Crew(
    agents=[data_analyst, development_factor_specialist, risk_adjustment_specialist, report_compiler],
    tasks=[task1, task2, task3, task4],
    verbose=True,
    process=Process.sequential
)

# Execute the analysis
result = crew.kickoff()

print("######################")
print(result)