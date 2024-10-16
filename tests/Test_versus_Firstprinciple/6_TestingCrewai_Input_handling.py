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
os.environ["OPENAI_API_KEY"] = "apikeyhere"
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
    description="You are an Actuary working with the following as data inputs {data}""",
    agent=actuarial_agent,
    expected_output="Return {data} .",
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
    result = crew.kickoff()
    
    print(result)

# Run the analysis
if __name__ == "__main__":
    run_chainladder_analysis()

###################################################################################################################################################################################################

