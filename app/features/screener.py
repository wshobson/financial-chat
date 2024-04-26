import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from finvizfinance.screener.overview import Overview

# Custom universe criteria, please see FinViz for all available filters
UNIVERSE_CRITERIA = {
    "Market Cap.": "+Small (over $300mln)",
    "Average Volume": "Over 750K",
    "Price": "Over $10",
    "Industry": "Stocks only (ex-Funds)",
    "20-Day Simple Moving Average": "Price above SMA20",
    "50-Day Simple Moving Average": "Price above SMA50",
    "200-Day Simple Moving Average": "SMA200 below SMA50",
    "EPS growththis year": "Over 20%",
    "EPS growthnext year": "Positive (>0%)",
    "EPS growthqtr over qtr": "Over 20%",
    "Sales growthqtr over qtr": "Over 20%",
    "Gross Margin": "Positive (>0%)",
    "Return on Equity": "Positive (>0%)",
}


def screener(filters):
    """
    Returns a dataframe of the screener view with the given filters, sorted by Market Cap.
    """
    view = Overview()
    view.set_filter(filters_dict=filters)
    df = view.screener_view(verbose=0)
    return df.sort_values(by="Market Cap", ascending=False)


def fetch_custom_universe():
    """
    Returns a custom universe of stocks based on the UNIVERSE_CRITERIA dictionary.
    """
    return screener(UNIVERSE_CRITERIA)


if __name__ == "__main__":
    from tabulate import tabulate

    print(tabulate(fetch_custom_universe(), headers="keys", tablefmt="psql"))
