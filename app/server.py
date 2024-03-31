import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langserve import add_routes
from openbb import obb

from app.chains.agent import agent_executor as agent_chain


OPENBB_API_TOKEN = os.environ.get("OPENBB_API_TOKEN")
obb.account.login(pat=OPENBB_API_TOKEN, remember_me=True)

app = FastAPI(
    title="Finance Chat 2000",
    version="1.0",
    description="The trading dude abides.",
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
    agent_chain,
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
