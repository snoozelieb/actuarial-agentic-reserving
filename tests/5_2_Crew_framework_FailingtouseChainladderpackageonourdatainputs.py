"""
Title: Lessons and cuntionalities I add in this version 

Human input:    Human input is needed to make tihs work.
                I follow the guidelines from the following docs:https://docs.crewai.com/how-to/Human-Input-on-Execution/
                Set human_input = True in tasks to allow for human input

Cache:          I also add cache functionality which was recommended by the crewai course.

Context:        I also add the context variable to the tasks to allow for the agents to have a better understanding of the task at hand. #https://docs.crewai.com/core-concepts/Tasks/#task-attributes

Creating your own tool:         https://docs.crewai.com/core-concepts/Tools/#creating-your-own-tools

Chain ladder package:           Inputs hould be data frames https://chainladder-python.readthedocs.io/en/latest/user_guide/triangle.html https://chainladder-python.readthedocs.io/en/latest/getting_started/tutorials/triangle-tutorial.html   
                                Data input should be a data frame end up with three columns: origin, development, values. 
                                    Origin and development should be datetime objects. 
                                    Origin should be the accident year I believe. Development year is the year that the payment is paid out.
                                    Values should be the amount of the claim. (paid, incurred)


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


import chainladder as cl
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
from langchain_community.tools import DuckDuckGoSearchRun

#Word document dependencies
from docx import Document
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE

#https://www.youtube.com/watch?v=q6QLGS306d0&ab_channel=TylerAI Crew ai tutorial that the code is based on
os.environ["OPENAI_API_KEY"] = "api key here"
os.environ["OPENAI_MODEL_NAME"]= "gpt-4o"

#############################################################################################################################################################################################################
#https://claude.ai/chat/dd07607b-4e11-47df-a012-01828a1f403e


from crewai import Task, Agent, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
import chainladder as cl
import pandas as pd
import numpy as np
from crewai_tools import SerperDevTool
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool, CSVSearchTool, CodeInterpreterTool
from langchain_core.tools import Tool as LangchainTool
from langchain_experimental.utilities import PythonREPL
from langchain.agents.agent_types import AgentType



# Set up the search tool
search_tool2 = Tool(
    name="Search",
    func=DuckDuckGoSearchRun().run,
    description="Useful for searching the web for chainladder documentation and usage examples."
)

# Initialize tools
docs_tool = DirectoryReadTool(directory='./blog-posts')
file_tool = FileReadTool()
search_tool = SerperDevTool()
csv_search_tool = CSVSearchTool()
codeinterpreter_tool = CodeInterpreterTool()

# Initialize Python REPL tool
python_repl = PythonREPL()
repl_tool = LangchainTool(
    name="python_repl",
    description="A Python shell for executing Python commands.",
    func=python_repl.run
)




# Define the Actuarial Analyst agent
actuarial_agent = Agent(
    role='Actuarial Analyst',
    goal='Automate chainladder analysis on insurance claims data',
    backstory='You are an experienced actuarial analyst proficient in using the chainladder package for claims reserving.',
    allow_delegation=False,
    verbose=True,
    tools=[repl_tool],
)

# Define the main task
# Define the main task with properly formatted expected output
chainladder_task = Task(
    description="Provide the results of the reserves based on the data inputs.""",
    agent=actuarial_agent,
    expected_output="Provide the results of the reserves based on the data inputs.",
)

# Create the crew
crew = Crew(
    agents=[actuarial_agent],
    tasks=[chainladder_task],
    verbose=True,
    process=Process.sequential
)


# Load sample data with corrected development periods
#Below is the format of the dataframe that is needed for the chainladder package
data = pd.DataFrame({
    'origin': [2015, 2015, 2015, 2016, 2016, 2016, 2017, 2017, 2017],
    'development': ['2015-12-31', '2016-12-31', '2017-12-31',
                    '2016-12-31', '2017-12-31', '2018-12-31',
                    '2017-12-31', '2018-12-31', '2019-12-31'],
    'values': [100, 150, 175, 110, 170, 200, 120, 180, 210]
})
"""
  Data input should be a data frame end up with three columns: origin, development, values. 
                                    Origin and development should be datetime objects. 
                                    Origin should be the accident year I believe. Development year is the year that the payment is paid out.
                                    Values should be the amount of the claim. (paid, incurred)
"""


# Main function to run the crew
def run_chainladder_analysis():
    # Generate sample data (replace this with your actual data loading process)
    raw_data = data
    
    # Run the crew
    result = crew.kickoff(inputs={
            "raw_data": raw_data.to_dict()})
    
    print(result)

# Run the analysis
if __name__ == "__main__":
    run_chainladder_analysis()

###################################################################################################################################################################################################




# import chainladder as cl
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt




# # Convert 'origin' and 'development' to datetime
# data['origin'] = pd.to_datetime(data['origin'].astype(str) + '-01-01')
# data['development'] = pd.to_datetime(data['development'])

# # Create a triangle from the raw data
# triangle = cl.Triangle(data, origin='origin', development='development', columns=['values'],cumulative=False)

# #Discplay the link-ratios
# print(triangle.link_ratio)

# #Vewing the latest diagonal
# print(triangle.latest_diagonal)

# # Display the triangle
# print("Run-off Triangle:")
# print(triangle)

# # Apply the chain ladder method
# cl_model = cl.Chainladder().fit(triangle)

# # Calculate Ultimate and Reserves
# ultimate = cl_model.ultimate_
# reserves = cl_model.full_expectation_ - triangle

# # Display results
# print("\nUltimate:")
# print(ultimate)

# print("\nReserves:")
# print(reserves)

# # Visualize the results
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# # Plot the original triangle
# triangle.plot(ax=ax1, title='Original Triangle')

# # Plot the completed triangle

# #understanding granularity
# print("origin grain:", triangle.origin_grain) #Y = yearly, M = monthly, Q = quarterly
# print("development grain:", triangle.development_grain) 


# plt.tight_layout()
# plt.show()


# ###################################################################################################################################################################################################
# """
# Non-crewai functions

# """
# from crewai_tools import BaseTool
# import chainladder as cl
# import pandas as pd
# import json

###################################################################################################################################################################################################
#######################################################################################################################################
#Reading in CSV file
# Function to read CSV file
# def read_csv_file(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         return df.to_json(orient='split')
#     except Exception as e:
#         return f"Error reading CSV file: {str(e)}"

# # Get CSV file path from user
# #csv_file_path = input("Enter the path to your actuarial data CSV file: ")

# # Read CSV file
# #csv_data = read_csv_file(csv_file_path)

# # Initialize tools
# docs_tool = DirectoryReadTool(directory='./blog-posts')
# file_tool = FileReadTool()
# search_tool = SerperDevTool()
# csv_search_tool = CSVSearchTool(r'C:\Users\masho\OneDrive\AI\Deloitte_framework\2_Analysis\Attempt4_Crewai\Triangles.csv')
# codeinterpreter_tool = CodeInterpreterTool()
# chainladder_tool = ChainladderTool()

# # Initialize Python REPL tool
# python_repl = PythonREPL()
# repl_tool = LangchainTool(
#     name="python_repl",
#     description="A Python shell for executing Python commands.",
#     func=python_repl.run
# )



# """
# Crewai specific functions for reserving based on run-off triangles
# """

# # Define agents
# development_factor_specialist = Agent(
#     role='Development Factor Specialist',
#     goal='Calculate development factors and apply the Chain Ladder method to estimate reserves',
#     backstory="""You are an experienced actuary specializing in calculating development factors for reserving. 
#     You are proficient in the Chain Ladder method and can provide best estimate, optimistic, and prudent reserve estimates.""",
#     verbose=True,
#     tools=[chainladder_tool, csv_search_tool],
#     allow_delegation=False,
#     cache=True,
# )

# report_compiler = Agent(
#     role='Report Compiler',
#     goal='Compile a comprehensive, regulation-compliant actuarial report',
#     backstory="""You are a meticulous report compiler with extensive knowledge of actuarial standards. 
#     You excel at integrating complex analyses into clear, compliant reports.""",
#     verbose=True,
#     tools=[csv_search_tool, chainladder_tool,],
#     allow_delegation=False,
#     cache=True,
# )

# # Define tasks
# task1 = Task(
#     description="""Using the provided run-off loss triangles, calculate development factors and apply the Chain Ladder method. 
#     Provide reserve estimates at best estimate (50th percentile), prudent (75th percentile), and optimistic (25th percentile).""",
#     agent=development_factor_specialist,
#     expected_output="Reserving analysis report with development factors and reserve estimates",
#     human_input=False,
# )

# task2 = Task(
#     description="""Compile a comprehensive actuarial report based on the Chain Ladder analysis. 
#     Include sections on methodologies, reserve calculations, and explanations for the best estimate, optimistic, and prudent estimates.
#     Ensure compliance with relevant actuarial standards.""",
#     agent=report_compiler,
#     expected_output="Final comprehensive actuarial report"
# )

# # Assemble the crew
# # crew = Crew(
# #     agents=[development_factor_specialist],
# #     tasks=[task1],
# #     verbose=True,
# #     process=Process.sequential,
# #)

# # Execute the analysis
# #result = crew.kickoff()

# #Addtional functionality to export results to a text file and word document 
# print("Results exported successfully!")

# print("######################")
# #print(result)