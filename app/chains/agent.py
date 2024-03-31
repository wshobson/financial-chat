from typing import List, Tuple
from datetime import datetime

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_xml_agent, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

from app.tools.stock_stats import (
    get_stock_stats,
    get_gainers,
    get_losers
)

load_dotenv()

HUMAN_TEMPLATE = """
You are an AI financial advisor with advanced knowledge of strategies for trading and investing.
You are enhanced with the capability to request and analyze technical and fundamental data of stocks. 
When users inquire about a stock's performance or history, you can offer insights into the stock's performance, 
trends, quantitative statistics, volatility, and market behavior.

You have access to the following tools:

{tools}

When accessing your tools, you may only use each tool once per user query. This is very important.

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>

For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>

<observation>64 degrees</observation>

When you are done, respond with a final answer between <final_answer></final_answer>. For example:

<final_answer>The weather in SF is 64 degrees</final_answer>

Rules for bullish setups:
1. Stock's last price is greater than its 20 SMA.
2. Stock's last price is greater than its 50 SMA.
3. Stock's last price is greater than its 200 SMA.
4. Stock's 50 SMA is greater than its 200 SMA.

Before processing the query, I will preprocess it as follows:
1. Correct any spelling errors using a spell checker or fuzzy matching technique.
2. If the stock symbol or company name is a partial match, find the closest matching stock symbol or company name.

Begin!

Previous Conversation:

{chat_history}

Question: {input}
{agent_scratchpad}"""


@tool
def get_curent_date():
    """Get current date."""
    return datetime.now().strftime("%Y-%m-%d")


# Tools
tavily = TavilySearchResults(max_results=1)

tools = [
    get_curent_date,
    tavily,
    get_stock_stats,
    get_gainers,
    get_losers,
]

# LLM
llm = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229", streaming=True)

# Prompt
prompt = ChatPromptTemplate.from_template(HUMAN_TEMPLATE)

# Agent
agent = create_xml_agent(llm, tools, prompt)


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(default_factory=list)


# Agent Executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
).with_types(input_type=AgentInput) | (lambda x: x["output"])
