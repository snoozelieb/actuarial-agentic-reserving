"""
The input you can give in context is either a valid dictionary or instance of Task "
In this version 2.5, I create a dictionary from the CSV file we get with 1 key and 1 value.

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


#######################################################################################################################################
def csv_to_dict_with_df(file_path):
    """
    Converts a CSV file into a dictionary with one key and a DataFrame as the value.
    
    Parameters:
        file_path (str): The path to the CSV file.
        key_name (str): The key name to use in the resulting dictionary.
        
    Returns:
        dict: A dictionary with the specified key and the DataFrame as the value.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Create the dictionary with one key and the DataFrame as the value
    result_dict = {'actuarial data': df}
    
    return result_dict

def csv_to_valid_list(file_path):
    """
    Converts a CSV file into a valid list format suitable for frameworks that expect
    a list rather than a dictionary or DataFrame as context input.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        list: A list of dictionaries where each dictionary represents a row of the DataFrame.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Convert the DataFrame to a list of dictionaries
    data_as_list = df.to_dict(orient='records')
    
    return data_as_list

def csv_to_valid_list_with_required_fields(file_path):
    """
    Converts a CSV file into a valid list format suitable for frameworks that expect a list,
    with a single dictionary containing the required keys like 'description', 'expected_output',
    and 'actuarial data' where the CSV content is stored in string format.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        list: A list containing a single dictionary with 'description', 'expected_output',
              and 'actuarial data' as keys and their respective values.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Convert the DataFrame to a string
    df_string = df.to_string(index=False)  # Converting the DataFrame to a string without including the index
    
    # Create a list with one dictionary containing the required keys
    result_list = [{
                'actuarial data': df_string
    }]
    
    return result_list

# Example usage
# file_path = 'path_to_your_csv_file.csv'
# key_name = 'actuarial_data'
# result_dict = csv_to_dict_with_df(file_path, key_name)

csv_data = csv_to_valid_list_with_required_fields(csv_file_path)

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
    Identify Lines of Business, assess data quality, and prepare the data for further analysis. 
    Provide a summary of your findings then create loss triangles based on the data.""",
    agent=data_analyst,
    context=csv_data,
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

print(result)
# Returns a TaskOutput object with the description and results of the task 1 (It works)
print(f"""
    Task completed!
    Task: {task1.output.description}
    Output: {task1.output.raw}
""")

print("######################")
