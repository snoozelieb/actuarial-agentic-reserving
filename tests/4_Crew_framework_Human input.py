"""
Title: Lessons and cuntionalities I add in this version 

Human input:    Human input is needed to make tihs work.
                I follow the guidelines from the following docs:https://docs.crewai.com/how-to/Human-Input-on-Execution/
                Set human_input = True in tasks to allow for human input

Cache:          I also add cache functionality which was recommended by the crewai course.

Context:        I also add the context variable to the tasks to allow for the agents to have a better understanding of the task at hand. #https://docs.crewai.com/core-concepts/Tasks/#task-attributes
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

#Word document dependencies
from docx import Document
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE

#https://www.youtube.com/watch?v=q6QLGS306d0&ab_channel=TylerAI Crew ai tutorial that the code is based on
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_MODEL_NAME"]= "gpt-4o"

#############################################################################################################################################################################################################
###################################################################################################################################################################################################
"""
Non-crewai functions

"""
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
#csv_file_path = input("Enter the path to your actuarial data CSV file: ")

# Read CSV file
#csv_data = read_csv_file(csv_file_path)

# Initialize tools
docs_tool = DirectoryReadTool(directory='./blog-posts')
file_tool = FileReadTool()
search_tool = SerperDevTool()
csv_search_tool = CSVSearchTool(r'C:\Users\masho\OneDrive\AI\Deloitte_framework\2_Analysis\Attempt4_Crewai\Triangles.csv')
codeinterpreter_tool = CodeInterpreterTool()

# Initialize Python REPL tool
python_repl = PythonREPL()
repl_tool = LangchainTool(
    name="python_repl",
    description="A Python shell for executing Python commands.",
    func=python_repl.run
)

#############################################################################################################################################################################################################
###################################################################################################################################################################################################

"""
Crewai specific functions for reserving based on run-off triangles
"""

# Define agents
development_factor_specialist = Agent(
    role='Development Factor Specialist',
    goal='Calculate development factors and apply the Chain Ladder method to estimate reserves',
    backstory="""You are an experienced actuary specializing in calculating development factors for reserving. 
    You are proficient in the Chain Ladder method and can provide best estimate, optimistic, and prudent reserve estimates.""",
    verbose=True,
    tools=[csv_search_tool, file_tool, repl_tool,],
    allow_delegation=False,
    cache=True,
)

report_compiler = Agent(
    role='Report Compiler',
    goal='Compile a comprehensive, regulation-compliant actuarial report',
    backstory="""You are a meticulous report compiler with extensive knowledge of actuarial standards. 
    You excel at integrating complex analyses into clear, compliant reports.""",
    verbose=True,
    tools=[csv_search_tool, file_tool, repl_tool, ],
    allow_delegation=False,
    cache=True,
)

# Define tasks
task1 = Task(
    description="""Using the provided run-off loss triangles, calculate development factors and apply the Chain Ladder method. 
    Provide reserve estimates at best estimate (50th percentile), prudent (75th percentile), and optimistic (25th percentile).""",
    agent=development_factor_specialist,
    expected_output="Reserving analysis report with development factors and reserve estimates",
    human_input=False,
)

task2 = Task(
    description="""Compile a comprehensive actuarial report based on the Chain Ladder analysis. 
    Include sections on methodologies, reserve calculations, and explanations for the best estimate, optimistic, and prudent estimates.
    Ensure compliance with relevant actuarial standards.""",
    agent=report_compiler,
    expected_output="Final comprehensive actuarial report"
)

# Assemble the crew
crew = Crew(
    agents=[development_factor_specialist],
    tasks=[task1],
    verbose=True,
    process=Process.sequential,
)

# Execute the analysis
result = crew.kickoff()

#Addtional functionality to export results to a text file and word document 
print("Results exported successfully!")

print("######################")
print(result)