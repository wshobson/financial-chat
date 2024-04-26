import pandas as pd


def wrap_dataframe(df: pd.DataFrame) -> str:
    df_string = df.to_markdown(index=False)
    return f"\n<observation>\n{df_string}\n</observation>\n"
