import pandas as pd


def wrap_dataframe(df: pd.DataFrame) -> str:
    df_string = df.to_markdown(index=False)
    return f"<observation>\n\n{df_string}\n\n</observation>"
