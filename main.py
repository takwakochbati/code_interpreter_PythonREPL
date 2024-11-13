from typing import Any
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor, AgentType
from langchain import hub
from langchain_core.tools import Tool
from langchain_experimental.agents import (
    create_csv_agent,
    create_pandas_dataframe_agent,
)
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama
import pandas as pd
from numpy.f2py.crackfortran import verbose

load_dotenv()


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        llm=ChatOllama(model="llama3.1", temperature=0),
        tools=tools,
    )
    python_agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    # python_agent_executor.invoke(
    #    input={
    #        "input": """generate a QRcode that points to https://github.com/takwakochbati, you have qrcode package installed already
    #                    and then create a folder in the current working directory and save the generated qrcode in it """
    #    }
    # )

    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOllama(model="llama3.1", temperature=0),
        path="episode_info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # csv_agent_executor.invoke(input={"input" : "how many columns are there in file episode_info.csv"})
    # csv_agent_executor.run("how many episodes are in season 4?")

    ############### *Create a router Agent that sends the work to either Python Agent or CSV Agent* ##############################
    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor.invoke,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                         takes an input the entire question and returns the answer after running pandas calculations""",
        ),
    ]
    prompt2 = base_prompt.partial(instructions="")
    router_agent = create_react_agent(
        prompt=prompt2,
        llm=ChatOllama(model="llama3.1", temperature=0),
        tools=tools,
    )
    router_agent_executor = AgentExecutor(router_agent, tools=tools, verbose=True)

    print(
        router_agent_executor.invoke(
            {
                "input": " which season has the most episodes?",
            }
        )
    )
    print(
        router_agent_executor.invoke(
            {
                "input": "Generate and save in current working directory 15 qrcodes that point to `www.udemy.com/course/langchain`",
            }
        )
    )


if __name__ == "__main__":
    main()
