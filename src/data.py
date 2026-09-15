import pandas as pd

def build_hawkes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate trades by timestamp into a binary directional signal (buy/sell)
    # and format the data as expected by the Hawkes model.

    # We go from Dataframe containing at least columns 'timestamp' and 'side' (is buy/sell)
    # We return a Dataframe with time_stamp (only unique values, determined by majority proportion of side)
    # And with a side = 1/0 and not buy/sell
    


    # Proportion of buys at each timestamp
    df_grouped = (
        df.assign(is_buy=(df["side"] == "buy").astype(int))
        .groupby("timestamp", as_index=False)["is_buy"]
        .mean()
    )

    # Majority vote: buy if proportion >= 50%
    df_grouped["side"] = (df_grouped["is_buy"] >= 0.5).astype(int)
    df_grouped = df_grouped.sort_values("timestamp").reset_index(drop=True)


    t = df_grouped["timestamp"].to_numpy() / 1e6 #convert to seconds
    t = t - t[0] #shift origin to 0
    side = df_grouped["side"].to_numpy() #side=1 is side=='buy', else 0

    df_hawkes = pd.DataFrame({"time_stamp": t, "side": side})

    return df_hawkes


def extract_time_window(t: np.ndarray, side: np.ndarray, start: float, duration: float) -> pd.DataFrame:
    
    # Extract a time window from the Hawkes-formatted data, with time re-based to 0.
    
    end = start + duration
    mask = (t >= start) & (t < end)

    t_win = t[mask] - t[mask][0]
    side_win = side[mask]

    return pd.DataFrame({"time_stamp": t_win, "side": side_win})