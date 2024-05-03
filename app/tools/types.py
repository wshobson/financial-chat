from langchain_core.pydantic_v1 import BaseModel, Field


class StockStatsInput(BaseModel):
    symbol: str = Field(..., description="The stock symbol to analyze")


class RMultipleInput(BaseModel):
    symbol: str = Field(..., description="The stock symbol to analyze")
    entry_price: float = Field(..., description="The entry price for the trade")
    stop_price: float = Field(..., description="The stop price for the trade")
    risk_multiple: int = Field(..., description="The R multiple for the profit target")


class PositionSizingInput(BaseModel):
    symbol: str = Field(..., description="The stock symbol to analyze")
    account_size: float = Field(..., description="The total account size in dollars")
    risk_percent: float = Field(
        ..., description="The percentage of the account to risk on the trade"
    )
    entry_price: float = Field(..., description="The entry price for the trade")
    stop_price: float = Field(..., description="The stop price for the trade")
