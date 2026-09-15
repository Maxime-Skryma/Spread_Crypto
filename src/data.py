import pandas as pd


from Hawkes_functions import (
    DHP,
    CHP, CHP_plot, CHP_IC, CHP_IC_plot,
    neg_log_likelihood, hawkes_residuals,
    neg_log_likelihood_bivariate, hawkes_residuals_bivariate,
    spectral_det, trace_1, trace_2,
    engle_russell_ed_test, _default_bivariate_bounds,
    fit_univariate, univariate_goodness_of_fit, pass_rate_univariate,
    simulate_bivariate, fit_bivariate, bivariate_goodness_of_fit,
    pass_rate_bivariate_sim,pass_rate_windows,fit_bivariate_sum_exp,neg_log_likelihood_bivariate_sum_exp,
    hawkes_residuals_bivariate_sum_exp,bivariate_goodness_of_fit_sum_exp,pass_rate_by_window_size,_eval_window_sum_exp
)

from scipy.optimize import minimize, NonlinearConstraint,LinearConstraint





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


def initialize_hawkes_params(t_win: np.ndarray, side_win: np.ndarray) -> tuple[np.ndarray, list, list]:

    duration = t_win[-1] - t_win[0]
    n_buy = side_win.sum()
    n_sell = len(side_win) - n_buy
    rate_buy, rate_sell = n_buy / duration, n_sell / duration
    dt_mean = np.mean(np.diff(t_win))

    theta_init = np.array([
        rate_buy * 0.3,         # mu1
        rate_sell * 0.3,        # mu2
        0.3,                    # R11
        0.1,                    # R21
        0.1,                    # R12
        0.3,                    # R22
        1.0 / (5 * dt_mean),    # beta1
        1.0 / (5 * dt_mean),    # beta2
    ])

    bounds = _default_bivariate_bounds()


    #We formalize constraints for trust-constraint

    constraints = [
        NonlinearConstraint(spectral_det, 1e-5, np.inf),
        NonlinearConstraint(trace_1, 1e-5, np.inf),
        NonlinearConstraint(trace_2, 1e-5, np.inf),
    ]

    return theta_init, bounds, constraints