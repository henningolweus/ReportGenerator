import pandas as pd


def twr_from_values_and_flows(values: pd.Series, flows: pd.Series) -> pd.Series:
    values = values.sort_index()
    flows = flows.reindex(values.index).fillna(0)
    daily_ret = (values - flows).div(values.shift(1)) - 1
    return daily_ret




