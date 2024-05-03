import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig
from openbb import obb
from dotenv import load_dotenv

import streamlit as st

from app.chains.clear_results import with_clear_container
from app.chains.agent import create_anthropic_agent_executor

load_dotenv()

obb.account.login(pat=os.environ.get("OPENBB_TOKEN"), remember_me=True)
obb.user.credentials.tiingo_token = os.environ.get("TIINGO_API_KEY")
obb.user.credentials.fmp_api_key = os.environ.get("FMP_API_KEY")
obb.user.credentials.intrinio_api_key = os.environ.get("INTRINIO_API_KEY")
obb.user.credentials.fred_api_key = os.environ.get("FRED_API_KEY")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

st.set_page_config(
    page_title="Financial Chat | The Trading Dude Abides",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = create_anthropic_agent_executor()

st.title("Financial Chat, your AI financial advisor 📈")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

with st.form(key="form"):
    user_input = st.text_input(
        "Please ask your question here:",
        placeholder="Please give me a full analysis of Apple",
    )
    submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()

    output_container.markdown(f"**User:** {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})

    answer_container = output_container.chat_message("assistant", avatar="💸")
    st_callback = StreamlitCallbackHandler(answer_container)

    cfg = RunnableConfig()
    cfg["callbacks"] = [st_callback]

    answer = st.session_state.agent_executor.invoke(
        {"input": user_input, "chat_history": st.session_state.messages}, cfg
    )

    if isinstance(answer, dict) and "output" in answer:
        st.session_state.messages.append(
            {"role": "assistant", "content": answer["output"]}
        )
        answer_container.markdown(f"**Assistant:** {answer['output']}")
