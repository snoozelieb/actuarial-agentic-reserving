
#######################################################################################################
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool, CSVSearchTool, CodeInterpreterTool
from langchain_core.tools import Tool as LangchainTool
from langchain_experimental.utilities import PythonREPL
from langchain.agents.agent_types import AgentType


#https://www.youtube.com/watch?v=q6QLGS306d0&ab_channel=TylerAI Crew ai tutorial that the code is based on
os.environ["OPENAI_API_KEY"] = "api-key-here"
os.environ["OPENAI_MODEL_NAME"]= "gpt-4o"


#######################################################################################################################################
#Reading in CSV file
# Function to read CSV file
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"

# Get CSV file path from user
csv_file_path = input("Enter the path to your actuarial data CSV file: ")

# Read CSV file
csv_data = read_csv_file(csv_file_path)

# Ensure data was read successfully
if csv_data is None:
    exit()


# Initialize tools
docs_tool = DirectoryReadTool(directory='./blog-posts')
file_tool = FileReadTool()
search_tool = SerperDevTool()
csv_search_tool = CSVSearchTool()
codeinterpreter_tool = CodeInterpreterTool()
#chainladder_tool = ChainladderTool()

# Initialize Python REPL tool
python_repl = PythonREPL()
repl_tool = LangchainTool(
    name="python_repl",
    description="A Python shell for executing Python commands.",
    func=python_repl.run
)


#######################################################################################################################################
# Define agents
data_analyst = Agent(
    role='Data Analyst',
    goal='Process and analyze actuarial data for multiple Lines of Business.You should identify data types, assess data quality, provide insights, and guide the user through the process of analyzing and splitting data by LOB.Ensure compliance with APN 401 Section 3 (Data) and consider user inputs for data-driven decisions',
    backstory="""You are a skilled data analyst specializing in actuarial data. 
    You excel at identifying data types, assessing data quality, and providing insights. You also excel at making run-off triangles from raw data and passing it to the development factor specialist.""",
    verbose=True,
    tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool,],
    allow_delegation=False,
    # You can pass an optional llm attribute specifying what model you wanna use.
    # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
)

development_factor_specialist = Agent(
    role='Development Factor Specialist',
    goal='Return a complete loss triangle. Calculate development factors and apply appropriate reserving methods',
    backstory="""You are an experienced actuary specializing in calculating development factors for reserving. 
    You are proficient in methods like Chain Ladder only.""",
    tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool,],
    verbose=True,
    allow_delegation=False,
    
)

risk_adjustment_specialist = Agent(
    role='Risk Adjustment Specialist',
    goal='Calculate risk adjustments using bootstrapping techniques',
    backstory="""You are an expert in calculating risk adjustments for insurance reserves. 
    You specialize in using bootstrapping techniques to calculate adjustments at the 75th percentile.""",
    verbose=True,
    tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool,],
    allow_delegation=False,
    
)

report_compiler = Agent(
    role='Report Compiler',
    goal='Compile a comprehensive, regulation-compliant actuarial report',
    backstory="""You are a meticulous report compiler with extensive knowledge of actuarial standards. 
    You excel at integrating complex analyses into clear, compliant reports.""",
    tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool,],
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
    The data is as follows: {csv_data}
    Identify Lines of Business, assess data quality, and prepare the data for further analysis. 
    Provide a summary of your findings then create loss triangles based on the data.""",
    agent=data_analyst,
    context=[csv_data],
    expected_output="Detailed data analysis report",
    human_input=False
)

task2 = Task(
    description="""Using this data, create loss triangles (paid and incurred claims trainagles),  calculate development factors and apply the Chain Ladder method only . 
    Provide reserve estimates at best estimate (50th percentile), prudent (75th percentile), and optimistic (25th percentile).""",
    agent=development_factor_specialist,
    context=[csv_data,task1],
    human_input=False,
    expected_output="Results of the loss triangles labeled with headings, and a Reserving analysis report with development factors and reserve estimates and a complete triangle"
)

task3 = Task(
    description="""Calculate risk adjustments using bootstrapping techniques at the 75th percentile Using the data and the context from earlier tasks. 
    Ensure alignment with IFRS 17 and APN 401 Section 6. 
    Document your methods, assumptions, and justification for margins.""",
    agent=risk_adjustment_specialist,
    context=[csv_data, task1, task2],
    human_input=False,
    expected_output="Risk adjustment analysis report with actual risk adjustment estimates and a dataframe with the risk adjustments"
)

task4 = Task(
    description="""Compile a comprehensive actuarial report integrating all previous analyses. 
    Include sections on data quality, methodologies, reserve calculations, and risk adjustments. 
    Ensure compliance with APN 401 and IFRS 17 standards.""",
    agent=report_compiler,
    context=[task1, task2, task3],
    human_input=False,
    expected_output="Final comprehensive actuarial report"
)

# Assemble the crew
crew = Crew(
    agents=[data_analyst, development_factor_specialist, risk_adjustment_specialist, report_compiler],
    tasks=[task1, task2, task3,],
    verbose=True,
    process=Process.sequential
)

# Execute the analysis
result = crew.kickoff()


# Returns a TaskOutput object with the description and results of the task 1 (It works)
print(f"""
    Task completed!
    Task: {task1.output.description}
    Output: {task1.output.raw}
""")

print("######################")
print(result)