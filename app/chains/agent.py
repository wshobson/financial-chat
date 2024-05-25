from typing import Annotated, TypedDict, Optional, Literal, Callable

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_node import tools_condition
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
from app.chains.templates import *

load_dotenv()


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "full_scan",
                "full_analysis",
                "chart_analysis",
                "determine_risk",
                "gainers_losers",
            ]
        ],
        update_dialog_stack,
    ]


class ToFullScanAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle a stock universe scan."""

    request: str = Field(
        description="Any necessary followup questions the scan assistant should clarify before proceeding."
    )


class ToFullAnalysisAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle a full analysis of a stock."""

    symbol: str = Field(description="The symbol of the stock to analyze.")
    request: str = Field(
        description="Any necessary followup questions the analysis assistant should clarify before proceeding."
    )


class ToChartAnalysisAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle a chart analysis of a stock."""

    symbol: str = Field(description="The symbol of the stock to analyze.")
    request: str = Field(
        description="Any necessary followup questions the chart assistant should clarify before proceeding."
    )


class ToRiskManagementAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle risk management."""

    symbol: str = Field(description="The symbol of the stock to analyze.")
    request: str = Field(
        description="Any necessary followup questions the risk assistant should clarify before proceeding."
    )


class ToGainersLosersAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle fetching gainers and losers."""

    request: str = Field(
        description="Any necessary followup questions the gainers/losers assistant should clarify before proceeding."
    )


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
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


def create_full_scan_agent(llm: Runnable) -> Assistant:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SCAN_TEMPLATE),
            ("placeholder", "{messages}"),
        ]
    )
    scan_tools = [get_stock_universe]
    runnable = prompt | llm.bind_tools(scan_tools)
    return Assistant(runnable)


def create_full_analysis_agent(llm: Runnable) -> Assistant:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FULL_ANALYSIS_TEMPLATE),
            ("placeholder", "{messages}"),
        ]
    )
    analysis_tools = [
        get_stock_price_history,
        get_key_metrics,
        get_stock_ratios,
        get_stock_sector_info,
        get_valuation_multiples,
        get_news_sentiment,
        get_relative_strength,
        get_stock_quantstats,
    ]
    runnable = prompt | llm.bind_tools(analysis_tools)
    return Assistant(runnable)


def create_chart_analysis_agent(llm: Runnable) -> Assistant:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CHART_ANALYSIS_TEMPLATE),
            ("placeholder", "{messages}"),
        ]
    )
    chart_tools = [get_stock_chart_analysis]
    runnable = prompt | llm.bind_tools(chart_tools)
    return Assistant(runnable)


def create_risk_management_agent(llm: Runnable) -> Assistant:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RISK_TEMPLATE),
            ("placeholder", "{messages}"),
        ]
    )
    risk_tools = [
        calculate_technical_stops,
        calculate_r_multiples,
        calculate_position_size,
    ]
    runnable = prompt | llm.bind_tools(risk_tools)
    return Assistant(runnable)


def create_gainers_losers_agent(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GAINERS_LOSERS_TEMPLATE),
            ("placeholder", "{messages}"),
        ]
    )
    runnable = prompt | llm.bind_tools([get_gainers, get_losers])
    return Assistant(runnable)


def create_primary_assistant(llm: Runnable) -> Assistant:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BASE_TEMPLATE),
            ("placeholder", "{messages}"),
        ]
    )
    tools = [TavilySearchResults(max_results=1)]
    runnable = prompt | llm.bind_tools(
        tools
        + [
            ToFullScanAssistant,
            ToFullAnalysisAssistant,
            ToChartAnalysisAssistant,
            ToRiskManagementAssistant,
            ToGainersLosersAssistant,
        ]
    )
    return Assistant(runnable)


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: AgentState) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and actions are not complete until after you have successfully invoked the appropriate tool."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


def should_continue(state: AgentState) -> Literal[
    "scan_stocks_tools",
    "analyze_stocks_tools",
    "chart_analysis_tools",
    "risk_management_tools",
    "gainers_losers_tools",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END

    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        tool_name = tool_calls[0]["name"]
        if tool_name in [
            "get_stock_price_history",
            "get_key_metrics",
            "get_stock_ratios",
            "get_stock_sector_info",
            "get_valuation_multiples",
            "get_news_sentiment",
            "get_relative_strength",
            "get_stock_quantstats",
        ]:
            return "analyze_stocks_tools"
        elif tool_name == "get_stock_universe":
            return "scan_stocks_tools"
        elif tool_name == "get_stock_chart_analysis":
            return "chart_analysis_tools"
        elif tool_name in [
            "calculate_technical_stops",
            "calculate_r_multiples",
            "calculate_position_size",
        ]:
            return "risk_management_tools"
        elif tool_name in ["get_gainers", "get_losers"]:
            return "gainers_losers_tools"
    return END


def route_primary_assistant(state: AgentState) -> Literal[
    "primary_assistant_tools",
    "enter_scan_stocks",
    "enter_analyze_stocks",
    "enter_chart_analysis",
    "enter_risk_management",
    "enter_gainers_losers",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END

    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFullScanAssistant.__name__:
            return "enter_scan_stocks"
        elif tool_calls[0]["name"] == ToFullAnalysisAssistant.__name__:
            return "enter_analyze_stocks"
        elif tool_calls[0]["name"] == ToChartAnalysisAssistant.__name__:
            return "enter_chart_analysis"
        elif tool_calls[0]["name"] == ToRiskManagementAssistant.__name__:
            return "enter_risk_management"
        elif tool_calls[0]["name"] == ToGainersLosersAssistant.__name__:
            return "enter_gainers_losers"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


def create_anthropic_agent_graph():
    llm = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")

    builder = StateGraph(AgentState)

    # Scan Assistant
    builder.add_node(
        "enter_scan_stocks",
        create_entry_node("Stock Scan Assistant", "scan_stocks"),
    )
    builder.add_node("scan_stocks", create_full_scan_agent(llm))
    builder.add_edge("enter_scan_stocks", "scan_stocks")
    builder.add_node(
        "scan_stocks_tools", create_tool_node_with_fallback([get_stock_universe])
    )
    builder.add_edge("scan_stocks_tools", "scan_stocks")
    builder.add_conditional_edges("scan_stocks", should_continue)

    # Analysis Assistant
    builder.add_node(
        "enter_analyze_stocks",
        create_entry_node("Stock Analysis Assistant", "analyze_stocks"),
    )
    builder.add_node("analyze_stocks", create_full_analysis_agent(llm))
    builder.add_edge("enter_analyze_stocks", "analyze_stocks")
    builder.add_node(
        "analyze_stocks_tools",
        create_tool_node_with_fallback(
            [
                get_stock_price_history,
                get_key_metrics,
                get_stock_ratios,
                get_stock_sector_info,
                get_valuation_multiples,
                get_news_sentiment,
                get_news_sentiment,
                get_relative_strength,
                get_stock_quantstats,
            ]
        ),
    )
    builder.add_edge("analyze_stocks_tools", "analyze_stocks")
    builder.add_conditional_edges("analyze_stocks", should_continue)

    # Chart Assistant
    builder.add_node(
        "enter_chart_analysis",
        create_entry_node("Stock Chart Analysis Assistant", "chart_analysis"),
    )
    builder.add_node("chart_analysis", create_chart_analysis_agent(llm))
    builder.add_edge("enter_chart_analysis", "chart_analysis")
    builder.add_node(
        "chart_analysis_tools",
        create_tool_node_with_fallback([get_stock_chart_analysis]),
    )
    builder.add_edge("chart_analysis_tools", "chart_analysis")
    builder.add_conditional_edges("chart_analysis", should_continue)

    # Risk Management Assistant
    builder.add_node(
        "enter_risk_management",
        create_entry_node("Stock Risk Management Assistant", "risk_management"),
    )
    builder.add_node("risk_management", create_risk_management_agent(llm))
    builder.add_edge("enter_risk_management", "risk_management")
    builder.add_node(
        "risk_management_tools",
        create_tool_node_with_fallback(
            [calculate_technical_stops, calculate_r_multiples, calculate_position_size]
        ),
    )
    builder.add_edge("risk_management_tools", "risk_management")
    builder.add_conditional_edges("risk_management", should_continue)

    # Gainers/Losers Assistant
    builder.add_node(
        "enter_gainers_losers",
        create_entry_node("Stock Gainers/Losers Assistant", "gainers_losers"),
    )
    builder.add_node("gainers_losers", create_gainers_losers_agent(llm))
    builder.add_edge("enter_gainers_losers", "gainers_losers")
    builder.add_node(
        "gainers_losers_tools",
        create_tool_node_with_fallback([get_gainers, get_losers]),
    )
    builder.add_edge("gainers_losers_tools", "gainers_losers")
    builder.add_conditional_edges("gainers_losers", should_continue)

    # Primary Assistant
    builder.add_node("primary_assistant", create_primary_assistant(llm))
    builder.add_node(
        "primary_assistant_tools",
        create_tool_node_with_fallback([TavilySearchResults(max_results=1)]),
    )
    builder.add_conditional_edges(
        "primary_assistant",
        route_primary_assistant,
        {
            "enter_scan_stocks": "enter_scan_stocks",
            "enter_analyze_stocks": "enter_analyze_stocks",
            "enter_chart_analysis": "enter_chart_analysis",
            "enter_risk_management": "enter_risk_management",
            "enter_gainers_losers": "enter_gainers_losers",
            "primary_assistant_tools": "primary_assistant_tools",
            END: END,
        },
    )
    builder.add_edge("primary_assistant_tools", "primary_assistant")
    builder.set_entry_point("primary_assistant")

    graph = builder.compile()
    return graph


if __name__ == "__main__":
    import json

    graph = create_anthropic_agent_graph()

    with open("graph_mermaid_image.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())

    with open("graph_output.json", "w") as f:
        json.dump(graph.get_graph().to_json(), f)
