import io
import os
import base64
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import pyimgur
import seaborn as sns

from dotenv import load_dotenv
from openbb import obb

from app.features.technical import add_technicals

load_dotenv()

IMGUR_CLIENT_ID = os.environ.get("IMGUR_CLIENT_ID")
IMGUR_CLIENT_SECRET = os.environ.get("IMGUR_CLIENT_SECRET")

sns.set_style("whitegrid")


def create_plotly_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Generate a Plotly chart for stock data visualization.

    This function creates a Plotly chart for a given DataFrame and stock symbol.
    It includes a candlestick chart with moving average overlays and subplots for RSI and ATR.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing stock data with columns like 'open', 'high', 'low', 'close', 'SMA_50', 'SMA_200', 'RSI', 'ATR'.
    - symbol (str): The stock symbol.

    Returns:
    - go.Figure: A Plotly figure object that can be used to display the chart.
    """
    fig = sp.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("", "", ""),
        row_heights=[0.6, 0.2, 0.2],
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            name="Price",
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        ),
        row=1,
        col=1,
    )

    # SMA_50 trace
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["SMA_50"], mode="lines", name="50-day SMA", line=dict(color="blue")
        ),
        row=1,
        col=1,
    )
    # SMA_200 trace
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["SMA_200"], mode="lines", name="200-day SMA", line=dict(color="red")
        ),
        row=1,
        col=1,
    )

    # RSI trace
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["RSI"], mode="lines", name="RSI", line=dict(color="orange")
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # ATR trace
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["ATR"], mode="lines", name="ATR", line=dict(color="orange")
        ),
        row=3,
        col=1,
    )

    now = datetime.now().strftime("%m/%d/%Y")
    fig.update_layout(
        height=600,
        width=800,
        title_text=f"{symbol} | {now}",
        title_y=0.98,
        plot_bgcolor="lightgray",
        xaxis_rangebreaks=[
            dict(bounds=["sat", "mon"], pattern="day of week"),
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(size=10)
    )

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="ATR", row=3, col=1)

    return fig


def upload_image_to_imgur(buffer, symbol) -> str:
    """
    Uploads an image to Imgur.

    This function takes a buffer containing an image, uploads it to Imgur, and returns the URL of the uploaded image.
    It uses the Imgur API credentials defined in the environment variables.

    Args:
        buffer (io.BytesIO): The buffer containing the image data.
        symbol (str): The stock symbol associated with the image, used for titling the image on Imgur.

    Returns:
        str: The URL of the uploaded image on Imgur.
    """
    im = pyimgur.Imgur(
        IMGUR_CLIENT_ID, client_secret=IMGUR_CLIENT_SECRET, refresh_token=True
    )

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tmp.write(buffer.getvalue())
        temp_path = tmp.name
        now = datetime.now().strftime("%m/%d/%Y")
        uploaded_image = im.upload_image(
            temp_path, title=f"{symbol} chart for {now}"
        )
        return uploaded_image.link


def plotly_fig_to_bytes(fig, filename="temp_plot.png"):
    """
    Convert a Plotly figure to a bytes object.

    This function takes a Plotly figure, saves it as a PNG image, reads the image back into memory,
    and then deletes the image file. It returns a bytes object of the image which can be used for further processing or uploading.

    Args:
        fig (plotly.graph_objs._figure.Figure): The Plotly figure to convert.
        filename (str): The filename to use when saving the image. Defaults to 'temp_plot.png'.

    Returns:
        io.BytesIO: A bytes object containing the image data.
    """
    fig.write_image(filename)
    with open(filename, "rb") as file:
        img_bytes = io.BytesIO(file.read())
    os.remove(filename)
    return img_bytes


def get_chart_base64(symbol: str) -> dict:
    """
    Generate a base64 encoded string of the chart image for a given stock symbol.
    Returns the base64 string and the figure object.

    Args:
    symbol (str): The stock symbol to generate the chart for.

    Returns:
    dict: A dictionary containing the base64 encoded string and the URL of the uploaded image on Imgur.
    """
    try:
        start = datetime.now() - timedelta(days=365 * 2)
        start_date = start.strftime("%Y-%m-%d")
        df = obb.equity.price.historical(
            symbol, start_date=start_date, provider="yfinance"
        ).to_df()

        if df.empty:
            return {"error": "Stock data not found"}

        df = add_technicals(df)
        chart_data = create_plotly_chart(df, symbol)
        chart_bytes = plotly_fig_to_bytes(chart_data)
        chart_url = upload_image_to_imgur(chart_bytes, symbol)

        chart_base64 = base64.b64encode(chart_bytes.getvalue()).decode('utf-8')
        return {"chart": chart_base64, "url": chart_url}
    except Exception as e:
        return {"error": f"Failed to generate chart: {str(e)}"}


