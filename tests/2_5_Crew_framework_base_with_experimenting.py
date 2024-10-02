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

# https://www.youtube.com/watch?v=q6QLGS306d0&ab_channel=TylerAI Crew ai tutorial that the code is based on
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4"

#######################################################################################################################################
# Reading in CSV file
# Function to read CSV file
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"

# Get CSV file path from user
#csv_file_path = input("Enter the path to your actuarial data CSV file: ")

# Read CSV file
#csv_data = read_csv_file(csv_file_path)

# Ensure data was read successfully
#if csv_data is None:
#        exit()

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

#######################################################################################################################################
# Define agents
lead_actuary = Agent(
    role='Lead Actuary',
    goal='Calculate actuarial claims reserving based on the data input provided. Coordinate with team members to complete the analysis and compile a comprehensive report with the best estimate claims reserve and bootsrapping to calculate the risk adjustment by calculating the 75th percentile claim reserve.',
    backstory='You are the lead actuary responsible for the overall reserving process. You will coordinate with team members to analyze data, calculate reserves, adjust for risk, and compile the final report.',
    verbose=True,
    llm = ChatAnthropic(model='claude-3-opus-20240229', temperature=0.8), # use this to specify a specific model like claude. Default is openai
    allow_delegation=False,
    tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool],
    allow_delegation=True
)

Actuarial_analyst = Agent(
    role='Data Analyst',
    goal='Calculate the claims reserve for the run-off triangle using the basic chain ladder method',
    backstory="""You are a skilled data analyst specializing in actuarial data. You excel at making run-off triangles and using the basic chain ladder method on the triangel, and passing on to the development factor specialist.""",
    verbose=True,
    tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool],
    allow_delegation=False
)

development_factor_specialist = Agent(
    role='Development Factor Specialist',
    goal='Return a complete loss triangle. Calculate development factors and apply appropriate reserving methods.',
    backstory="""You are an experienced actuary specializing in calculating development factors for reserving. 
    You are proficient in methods like Chain Ladder only.""",
    tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool],
    verbose=True,
    allow_delegation=False
)

risk_adjustment_specialist = Agent(
    role='Risk Adjustment Specialist',
    goal='Calculate risk adjustments using bootstrapping techniques.',
    backstory="""You are an expert in calculating risk adjustments for insurance reserves. 
    You specialize in using bootstrapping techniques to calculate adjustments at the 75th percentile.""",
    verbose=True,
    tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool],
    allow_delegation=True
)

report_compiler = Agent(
    role='Report Compiler',
    goal='Compile a comprehensive, regulation-compliant actuarial report.',
    backstory="""You are a meticulous report compiler with extensive knowledge of actuarial standards. 
    You excel at integrating complex analyses into clear, compliant reports.""",
    tools=[csv_search_tool, file_tool, repl_tool, codeinterpreter_tool],
    verbose=True,
    allow_delegation=True
)

#######################################################################################################################################
# Define the single task
reserving_task = Task(
    description=f"""Calculate actuarial claims reserving based on the data input provided. 
    You should analyze the data, calculate development factors, apply reserving methods, calculate risk adjustments, and compile a comprehensive, regulation-compliant actuarial report.""",
    agent=lead_actuary,
    expected_output="Comprehensive actuarial reserving calculation and report",
    human_input=False
)

# Assemble the crew
crew = Crew(
    agents=[lead_actuary],
    tasks=[reserving_task],
    verbose=True,
    process=Process.sequential  # You can change this to Process.parallel if you prefer
)

# Execute the analysis
result = crew.kickoff()

# Returns a TaskOutput object with the description and results of the task
print(f"""
    Task completed!
    Task: {reserving_task.output.description}
    Output: {reserving_task.output.raw}
""")

print("######################")
print(result)