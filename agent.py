import os
os.environ["LANGCHAIN_ALLOW_DANGEROUS_TOOLS"] = "true"

import re
import pandas as pd
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Convert Google Sheet URL to export CSV URL
def convert_gsheet_url_to_csv(url: str) -> str | None:
    pattern = r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)(?:/.*?gid=([0-9]+))?"
    match = re.search(pattern, url)
    if not match:
        return None
    spreadsheet_id = match.group(1)
    gid = match.group(2) if match.group(2) else "0"
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"

# ✅ Load selected Google Sheets into a combined DataFrame
def load_sheets_to_df(selected_sheets: list, sheets_dict: dict) -> pd.DataFrame | None:
    if not selected_sheets:
        return None

    dfs = []
    for name in selected_sheets:
        try:
            df_part = pd.read_csv(sheets_dict[name])
            df_part["_source_sheet"] = name
            dfs.append(df_part)
        except Exception as e:
            print(f"⚠️ Error loading {name}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

# ✅ Run the user query using LangChain agent with tools
def run_query_with_agent(df: pd.DataFrame, query: str) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

    pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    pandas_tool = Tool.from_function(
        name="Pandas DataFrame Agent",
        func=pandas_agent.run,
        description="Useful for answering questions about the uploaded data."
    )

    python_tool = Tool.from_function(
        name="Python Interpreter",
        func=PythonREPL(allow_dangerous_code=True).run,
        description="Useful for Python calculations or plotting."
    )

    agent = initialize_agent(
        tools=[pandas_tool, python_tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
    )

    full_response = agent.run(query)

    # Extract any code from the answer (if available)
    code_match = re.search(r"```python(.*?)```", full_response, re.DOTALL)
    code = code_match.group(1).strip() if code_match else None

    return {
        "answer": full_response,
        "code": code
    }
