from typing import List, Tuple

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

from app.tools.stock_stats import (
    get_gainers,
    get_losers,
    get_stock_price_history,
    get_stock_quantstats,
    get_stock_ratios,
    get_key_metrics,
    get_stock_sector_info,
    get_valuation_multiples,
)
from app.tools.stock_sentiment import get_news_sentiment
from app.tools.stock_relative_strength import get_relative_strength

load_dotenv()

SYSTEM_TEMPLATE = """
You are a specialized financial advisor with advanced knowledge of trading, investing, quantitative finance, technical analysis, and fundamental analysis.

You should never perform any math on your own, but rather use the tools available to you to perform all calculations.

CRITERIA FOR BULLISH SETUPS:
----------------------------

You will find below the criteria to use for classification of bullish setups in the stock market. Any trading setups should 
be based off the daily timeframe and the most recent data.

Rules for bullish setups based on the stock's most recent closing price:
1. Stock's closing price is greater than its 20 SMA.
2. Stock's closing price is greater than its 50 SMA.
3. Stock's closing price is greater than its 200 SMA.
4. Stock's 50 SMA is greater than its 150 SMA.
5. Stock's 150 SMA is greater than its 200 SMA.
6. Stock's 200 SMA is trending up for at least 1 month.
7. Stock's closing price is at least 30 percent above 52-week low.
8. Stock's closing price is within 25 percent of its 52-week high.
9. Stock's 30-day average volume is greater than 750K.
10. Stock's ADR percent is less than 5 percent and greater than 1 percent.
11. Stock's trendline slope is positive and rising.
12. Stock's relative strength rank is above 80.

PREPROCESSING:
--------------

Before processing the query, you will preprocess it as follows:
1. Correct any spelling errors using a spell checker or fuzzy matching technique.
2. If the stock symbol or company name is a partial match, find the closest matching stock symbol or company name."""


def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("user", SYSTEM_TEMPLATE),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


def get_tools(llm):
    tavily = TavilySearchResults(max_results=1)

    tools = [
        get_relative_strength,
        get_valuation_multiples,
        get_stock_price_history,
        get_stock_quantstats,
        get_gainers,
        get_losers,
        get_stock_sector_info,
        get_news_sentiment,
        get_stock_ratios,
        get_key_metrics,
        tavily,
    ]
    return tools


def create_anthropic_agent_executor():
    llm = ChatAnthropic(
        temperature=0,
        model_name="claude-3-opus-20240229",
        max_tokens=4096,
    )

    tools = get_tools(llm)
    prompt = get_prompt()

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})


def get_anthropic_agent_executor_chain():
    executor = create_anthropic_agent_executor()
    return executor.with_types(input_type=AgentInput) | (lambda x: x["output"])
