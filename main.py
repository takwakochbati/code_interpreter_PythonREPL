from langchain_community.document_loaders import ReadTheDocsLoader
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor, AgentType
from langchain import hub
from langchain_experimental.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama
import pandas as pd


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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    agent_executor.invoke(
        input={
            "input": """generate a QRcode that points to https://github.com/takwakochbati, you have qrcode package installed already
                        and then create a folder in the current working directory and save the generated qrcode in it """
        }
    )

    csv_agent = create_csv_agent(
        llm=ChatOllama(model="llama3.1", temperature=0),
        path="episode_info.csv",
        verbose = True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    csv_agent.invoke(input={"input" : "how many columns are there in file episode_info.csv"})
    csv_agent.run("how many episodes are in season 4?")

if __name__ == "__main__":
    main()
