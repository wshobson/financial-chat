from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_node import tools_condition
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import AnyMessage, add_messages
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
    get_stock_universe,
)
from app.tools.stock_sentiment import get_news_sentiment
from app.tools.stock_relative_strength import get_relative_strength
from app.tools.stock_charts import get_stock_chart_analysis
from app.tools.risk_management import (
    calculate_r_multiples,
    calculate_technical_stops,
    calculate_position_size,
)
from app.tools.utils import create_tool_node_with_fallback
from app.chains.templates import SYSTEM_TEMPLATE

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: AgentState, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            ("placeholder", "{messages}"),
        ]
    )


def get_tools():
    tavily = TavilySearchResults(max_results=1)

    tools = [
        get_stock_universe,
        get_stock_chart_analysis,
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
        calculate_r_multiples,
        calculate_technical_stops,
        calculate_position_size,
    ]
    return tools


def create_anthropic_agent_graph():
    llm = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")

    tools = get_tools()
    prompt = get_prompt()

    llm_with_tools = prompt | llm.bind_tools(tools)

    builder = StateGraph(AgentState)

    builder.add_node("assistant", Assistant(llm_with_tools))
    builder.add_node("action", create_tool_node_with_fallback(tools))

    builder.set_entry_point("assistant")

    builder.add_conditional_edges(
        "assistant",
        tools_condition,
        {"action": "action", END: END},
    )
    builder.add_edge("action", "assistant")

    graph = builder.compile()
    return graph
