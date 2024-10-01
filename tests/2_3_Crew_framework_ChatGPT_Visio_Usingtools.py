"""
Goal is to make CSVsearch
"""
import pandas as pd
import numpy as np
import chainladder as cl

# Recreating the given loss triangle data from the user's input
data = {
    "2018": [268620, 266270, np.nan, np.nan, np.nan, np.nan],
    "2019": [277470, 465371, np.nan, np.nan, np.nan, np.nan],
    "2020": [365103, 438107, np.nan, np.nan, np.nan, np.nan],
    "2021": [332407, 318382, 6541, np.nan, np.nan, np.nan],
    "2022": [360737, 327925, np.nan, np.nan, np.nan, np.nan],
    "2023": [272761, np.nan, np.nan, np.nan, np.nan, np.nan]
}

# Converting the data into a pandas DataFrame
triangle_df = pd.DataFrame(data).T
triangle_df.columns = ["Development Year 1", "Development Year 2", "Development Year 3", "Development Year 4", "Development Year 5", "Development Year 6"]

# Creating a loss triangle object using chainladder
triangle = cl.Triangle(triangle_df)

# Fitting the chainladder model to estimate the reserve
chainladder_model = cl.Chainladder().fit(triangle)
reserve_estimate = chainladder_model.reserve_.sum().values[0]

reserve_estimate




# This script demonstrates the use of the Crew framework in CrewAI to form claims triangles and analyze actuarial data.
import os
os.environ["OPENAI_API_KEY"] = "api-key-here"
os.environ["OPENAI_MODEL_NAME"]= "gpt-4o"

import pandas as pd 
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import chainladder as cl  # Added import for chainladder
from crewai import Agent, Task, Crew, Process
import logging
logging.basicConfig(level=logging.DEBUG)

#############################################################################################################################################################################################################
# Importing CrewAI tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    CSVSearchTool, CodeInterpreterTool,
)

# Instantiate tools
docs_tool = DirectoryReadTool(directory='./blog-posts')
file_tool = FileReadTool()
search_tool = SerperDevTool()
csv_search_tool = CSVSearchTool(r'C:\Users\masho\OneDrive\AI\Deloitte_framework\2_Analysis\Attempt4_Crewai\sample_motor_claims_data.csv')  #https://github.com/alexfazio/crewAI-quickstart/blob/main/crewai_sequential_CSVSearchTool_quickstart.ipynb

#codeinterpreter_tool = CodeInterpreterTool()

###################################################################################################################################################################################################
"""
Non-crewai functions
"""

# Reading in CSV file
# Function to read CSV file
#def read_csv_file(file_path):
#    """
#    Reads the CSV file at the specified file path and converts it to a DataFrame.
#    """
#    try:
#        df = pd.read_csv(file_path)
#        return df
#    except Exception as e:
#        return f"Error reading CSV file: {str(e)}"

# Get CSV file path from user
#csv_file_path = input("Enter the path to your actuarial data CSV file: ")

# Read CSV file
#csv_data = read_csv_file(csv_file_path)

#############################################################################################################################################################################################################
###################################################################################################################################################################################################

"""
CrewAI specific functions
"""

#########################################################################################################################################################################################
"""
TOOLS
"""



"""
Langchain toolkit importing
"""

# Importing LangChain tools for interaction with Python shell and dataframes
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL 

# Instantiate Python shell tool
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# Importing tools for agents to interact with a pandas data frame
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Load data into pandas dataframe
#df = pd.read_csv(csv_file_path)

# Extract the list of unique LOBs
#LOB_list = df['LOB'].unique()

# Prepare inputs for each LOB
#inputs = []
#for lob in LOB_list:
#    df_lob = df[df['LOB'] == lob]
#    inputs.append({'LOB': lob, 'data': df_lob})


#########################################################################################################################################################################################
"""
ChainLadder Function and Tool Integration
"""

# # Define function to calculate chainladder reserve
# def calculate_chainladder_reserve(data):
#     """
#     Calculates reserve estimates using the ChainLadder method.

#     Parameters:
#         data (pd.DataFrame): A DataFrame representing the loss triangle.

#     Returns:
#         float: The estimated reserve.
#     """
#     try:
#         triangle = cl.Triangle(data)
#         model = cl.Chainladder().fit(triangle)
#         reserve_estimate = model.reserve_.sum().values[0]
#         return reserve_estimate
#     except Exception as e:
#         return f"An error occurred: {e}"

# # Create a tool for the ChainLadder function
# chainladder_tool = Tool(
#     name="calculate_chainladder_reserve",
#     func=calculate_chainladder_reserve,
#     description="Calculates reserve estimates using the ChainLadder method.",
# )

######################################################################################################################################################################################################
# Define agents

# Define the Information Gatherer agent
information_gatherer = Agent(
    role='Information Gatherer',
    goal='Gather and clarify essential information for actuarial claims reserving from the provided CSV file',
    backstory="""You are a meticulous information gatherer with expertise in actuarial data. 
    Your role is to ensure all necessary details are collected and verified before the actuarial analysis begins.""",
    tools=[csv_search_tool, docs_tool, search_tool, file_tool], 
    verbose=True,
    allow_delegation=False,
    cache=False
)

data_analyst = Agent(
    role='Data Analyst',
    goal='Process and analyze actuarial data for multiple Lines of Business from the provided CSV file.',
    backstory="""You excel at identifying data types, assessing data quality, and providing insights, creating run-off triangles from raw data.""",
    tools=[csv_search_tool, docs_tool, search_tool, file_tool],
    verbose=True,
    # llm = ChatAnthropic(model='claude-3-opus-20240229', temperature=0.8), # use this to specify a specific model like claude. Default is openai
    allow_delegation=False,
    cache=False
)

development_factor_specialist = Agent(
    role='Development Factor Specialist',
    goal='Calculate the run-off triangles and calculate development factors',
    backstory="""You specialize in calculating development factors for reserving, primarily using the Chain Ladder method.""",
    #tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool, chainladder_tool],  # Added chainladder_tool
    verbose=True,
    tools=[csv_search_tool, docs_tool, search_tool, file_tool],
    allow_delegation=False,
    cache=False
)

risk_adjustment_specialist = Agent(
    role='Risk Adjustment Specialist',
    goal='Calculate risk adjustments using bootstrapping techniques',
    backstory="""You specialize in using bootstrapping techniques to calculate adjustments at the 75th percentile.""",
    #tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool, chainladder_tool],  # Added chainladder_tool
    verbose=True,
    allow_delegation=False,
    cache=False,
)

report_compiler = Agent(
    role='Report Compiler',
    goal='Compile a comprehensive, regulation-compliant actuarial report',
    backstory="""You excel at integrating complex analyses into clear, compliant reports.""",
    #tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool], 
    verbose=True,
    allow_delegation=False,
    cache=False,
)

#######################################################################################################################################
# Define tasks

# Task to gather and organize LOB data into a dictionary
task_gather_info = Task(
    description=f"""Analyze the provided actuarial data and gather essential information for claims reserving. 
    Create a dictionary with LOB as keys and claim data as values.""",
    agent=information_gatherer,
    expected_output="Dictionary of LOB with associated claims data",
    human_input=False
)

# Task to create run-off triangles
task_form_triangles = Task(
    description="""Create run-off triangles (Paid and Incurred) by LOB from the dictionary data. 
    Ensure accurate data and prepare for reserving analysis.""",
    agent=data_analyst,
    context=[task_gather_info],
    expected_output="Run-off triangle data",
    #Output_file=r'C:\Users\masho\OneDrive\AI\Deloitte_framework\2_Analysis\Attempt4_Crewai\Data_analyst_output_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.md',
    human_input=False
)

# Task to calculate reserves using the Chain Ladder method
task_calculate_reserves = Task(
    description="""Using the run-off triangles, calculate the Actuarial reserve using the Chain Ladder method for each LOB. 
    Provide best estimate, prudent, and optimistic reserve estimates.""",
    agent=development_factor_specialist,
    context=[task_gather_info, task_form_triangles],
    expected_output="Reserve calculations report",
    human_input=True
)

# Task to calculate risk adjustments
task_calculate_risk_adjustment = Task(
    description="""Using the calculated reserves, perform risk adjustment calculations using bootstrapping techniques at the 75th percentile.""",
    agent=risk_adjustment_specialist,
    context=[task_gather_info, task_form_triangles, task_calculate_reserves],
    expected_output="Risk adjustment analysis report"
)

# Compile the final actuarial report
task_compile_report = Task(
    description="""Compile a comprehensive actuarial report integrating all analyses: data quality, methodologies, reserve calculations, and risk adjustments.""",
    agent=report_compiler,
    expected_output="Final actuarial report"
)

######################################################################################################################################################################################################
# Assemble the crew to process multiple LOBs using "Kickoff for each"

# Initialize Crew with the agents and tasks, iterating through each LOB using kickoff for each
crew = Crew(
    agents=[information_gatherer, data_analyst, development_factor_specialist, risk_adjustment_specialist, report_compiler],
    tasks=[task_gather_info, task_form_triangles, task_calculate_reserves],
    verbose=True,
    process=Process.sequential
)

# Execute the analysis with "kickoff for each" to handle multiple LOBs
result = crew.kickoff()

# Output and handle the final result
print("Results exported successfully!")
print("######################")
#print(result)

"""
Explanation:

Idea 1 Implementation:

Agents:
data_analyst_crew1 and data_analyst_crew2 form the claims triangles independently.
comparison_agent compares the results and investigates discrepancies.
Tasks:
task_form_triangles_crew1 and task_form_triangles_crew2 are assigned to the respective crews.
task_compare_results is assigned to the comparison agent.
Crew Execution:
crew_main is assembled with the agents and tasks for Idea 1.
The kickoff method executes the tasks sequentially.
Idea 2 Implementation:

Data Preparation:
The data is split into a dictionary lob_data_dict with LOBs as keys.
Agents and Tasks:
For each LOB, an agent and task are dynamically created using create_lob_agent and task_analyze_lob.
Crew Execution:
For each LOB, a crew crew_lob is assembled and the kickoff method is called to analyze the data and calculate reserves.
Tools Used:

Included csv_search_tool, file_tool, repl_tool, and codeinterpreter_tool for the agents.
Used the PythonREPL and pandas agents to interact with the data.
Data Input:

The CSV data is read and converted to JSON format for the agents to process.
Notes:

Make sure to have the necessary Crew AI and LangChain libraries installed.
Adjust the paths and imports based on your environment.
Replace placeholders and adjust the code as needed to fit your specific use case.
Additional Steps:

Exporting Results:
You can add functionality to export the results to files if needed.
Error Handling:
Ensure to include error handling for cases where data might be missing or malformed.
Testing:
Test the code with sample data to ensure it works as expected.
Final Output:

The code will process each LOB separately and print the results.
It will also perform the main analysis where two crews form triangles and a third crew compares them.


Key updates made:
Iteration through LOBs: Implemented "kickoff for each" to process each Line of Business iteratively.
Task organization: Adjusted tasks according to the steps indicated in the workflow.
Human input points: Set up human input where necessary based on the process described in the workflow.
Detailed comments: Added comments to each part of the code to clarify the functionality and purpose of each section.

Version 2_2_Crew_framework_ChatGPT_Visio_Chainladderpackage.py
https://chatgpt.com/c/66f84aa9-87f0-8008-855d-afa95ceead1c

. The key updates include:

Importing the chainladder package.
Defining the calculate_chainladder_reserve function that uses chainladder.
Creating a chainladder_tool and registering it within your CrewAI framework.
Adding the chainladder_tool to the development_factor_specialist agent.
"""