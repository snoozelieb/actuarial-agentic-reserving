
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
os.environ["OPENAI_API_KEY"] = "api-key-here"
os.environ["OPENAI_MODEL_NAME"]= "gpt-4o"


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
    You excel at identifying data types, assessing data quality, and providing insights. You also excel at making run-off triangles from raw data and passing it to the development factor specialist.""",
    verbose=True,
    allow_delegation=False,
    # You can pass an optional llm attribute specifying what model you wanna use.
    # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
)

development_factor_specialist = Agent(
    role='Development Factor Specialist',
    goal='Review the accuracy of the run off triangles created based on input data. Calculate development factors and apply appropriate reserving methods',
    backstory="""You are an experienced actuary specializing in calculating development factors for reserving. 
    You are proficient in methods like Chain Ladder only.""",
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
    allow_delegation=False,
    
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


# Returns a TaskOutput object with the description and results of the task
print(f"""
    Task completed!
    Task: {task1.output.description}
    Output: {task1.output.raw}
""")

print("######################")
#print(result)