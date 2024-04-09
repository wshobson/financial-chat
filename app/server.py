import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langserve import add_routes
from openbb import obb
from dotenv import load_dotenv


from app.chains.agent import get_anthropic_agent_executor_chain

load_dotenv()

OPENBB_TOKEN = os.environ.get("OPENBB_TOKEN")
obb.account.login(pat=OPENBB_TOKEN)
obb.user.credentials.tiingo_token = os.environ.get("TIINGO_API_KEY")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

app = FastAPI(
    title="Financial Chat",
    version="1.0",
    description="The Trading Dude Abides",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(
    app,
    get_anthropic_agent_executor_chain(),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
