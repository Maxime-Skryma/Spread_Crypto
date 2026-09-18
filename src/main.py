import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model import (
    fit_bivariate, bivariate_goodness_of_fit,
    fit_bivariate_sum_exp, bivariate_goodness_of_fit_sum_exp,
    compare_kernels_by_window_size, qq_overlay,
    plot_params_across_day,
)
from data import (
    build_hawkes_dataframe, initialize_hawkes_params,
    initialize_hawkes_params_sum_exp
)


def plot_price_volatility(raw_df, price_col='price', freq='1min', vol_window=30):
    d = raw_df.copy()
    d['dt'] = pd.to_datetime(d['timestamp'], unit='us', utc=True)
    d = d.set_index('dt').sort_index()
    price = d[price_col].resample(freq).last().ffill()
    vol = np.log(price).diff().rolling(vol_window).std()
    h = (price.index - price.index[0]).total_seconds() / 3600.0

    fig, ax1 = plt.subplots(figsize=(14, 3))
    ax1.plot(h, price.values, color='navy', lw=0.7, label='Price')
    ax1.set_ylabel('Price (USD)'); ax1.set_xlabel('hours')
    ax2 = ax1.twinx(); ax2.plot(h, vol.values, color='deepskyblue', lw=0.7, label='Volatility')
    ax2.set_ylabel('Std returns horaires')
    ax1.set_title('Price & Volatility'); plt.tight_layout(); plt.show()


def run_real_data(
    raw_df,
    out_dir="results",
    window_start="2026-09-01 13:05:00",
    window_end="2026-09-01 13:10:00",
    win_sizes=(5, 10, 20, 60),                 # size of windows : but it will be tested on all day
    n_windows=10,
    seed=42,
    roll_sizes=(10,),               
    roll_min_points=30,                
):
    os.makedirs(out_dir, exist_ok=True)

    counter = [1]
    original_show = plt.show

    def save_and_close(*args, **kwargs):
        plt.savefig(f"{out_dir}/fig_{counter[0]:02d}.png", bbox_inches='tight')
        counter[0] += 1
        plt.close()
    plt.show = save_and_close

    try:
        with open(f"{out_dir}/results_summary.txt", "w") as f:
            f.write("=== REAL DATA RESULTS ===\n")
            f.write(f"Window : {window_start} -> {window_end}\n\n")

            ts = pd.to_datetime(raw_df["timestamp"], unit="us", utc=True)
            start = pd.to_datetime(window_start, utc=True)
            end = pd.to_datetime(window_end, utc=True)
            raw_win = raw_df[(ts >= start) & (ts < end)]

            if len(raw_win) == 0:
                raise ValueError("Window empty : check dates and hours")

            # Preprocessing only on the window
            df_hawkes = build_hawkes_dataframe(raw_win)
            t_real = df_hawkes['time_stamp'].to_numpy()
            side_real = df_hawkes['side'].to_numpy()
            f.write(f"Events in window : {len(df_hawkes)} "
                    f"(buys={int(side_real.sum())}, sells={int(len(side_real) - side_real.sum())})\n\n")

            # Fit (Mono-Exp)
            theta_init, bounds, constraints = initialize_hawkes_params(t_real, side_real)
            res_biv = fit_bivariate(
                df_hawkes,
                theta_init=theta_init,
                bounds=bounds,
                constraints=constraints,
                verbose=False
            )

            # Residuals (bivariate)
            (u1_mono, u2_mono), gof_biv = bivariate_goodness_of_fit(res_biv.x, df_hawkes, verbose=False)

            f.write("--- Global Bivariate Fit (Mono-Exp) ---\n")
            f.write(f"Dim 1 (Buy)  -> KS: {gof_biv[0][0]:.4f} | LB: {gof_biv[0][1]:.4f} | ED: {gof_biv[0][2]:.4f}\n")
            f.write(f"Dim 2 (Sell) -> KS: {gof_biv[1][0]:.4f} | LB: {gof_biv[1][1]:.4f} | ED: {gof_biv[1][2]:.4f}\n\n")

            # 3. Fit Bivariate (Sum-Exp) and figures
            theta_init_sum = initialize_hawkes_params_sum_exp(t_real, side_real)

            # FIX: bounds adapted from time-scale data
            inv_dt = 1.0 / np.mean(np.diff(t_real))
            bounds_sum = (
                (1e-5, None), (1e-5, None),
                (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                (1e-3, inv_dt), (1e-3, inv_dt),
                (inv_dt, 1000.0 * inv_dt), (inv_dt, 1000.0 * inv_dt),
            )

            res_sum = fit_bivariate_sum_exp(
                df_hawkes,
                theta_init=theta_init_sum,
                bounds_sum=bounds_sum,
                verbose=False
            )

            (u1_sum, u2_sum), gof_sum = bivariate_goodness_of_fit_sum_exp(res_sum.x, df_hawkes, verbose=False)

            f.write("--- Global Bivariate Fit (Sum-Exp) ---\n")
            f.write(f"Dim 1 (Buy)  -> KS: {gof_sum[0][0]:.4f} | LB: {gof_sum[0][1]:.4f} | ED: {gof_sum[0][2]:.4f}\n")
            f.write(f"Dim 2 (Sell) -> KS: {gof_sum[1][0]:.4f} | LB: {gof_sum[1][1]:.4f} | ED: {gof_sum[1][2]:.4f}\n\n")

            # PDF
            qq_overlay(u1_mono, u1_sum, title="Dimension 1 (Buys)")
            qq_overlay(u2_mono, u2_sum, title="Dimension 2 (Sells)")

            # Windows
            f.write("--- Window Comparison (sampled across the full day) ---\n")
            f.write("Per-test pass rate (KS/LB/ED) + mean test statistics, by window size\n")
            f.write("-" * 60 + "\n")

            # Windows comparison on all the day
            # QQ-plot only on the window we defined
            df_hawkes_full = build_hawkes_dataframe(raw_df)
            t_full = df_hawkes_full['time_stamp'].to_numpy()
            side_full = df_hawkes_full['side'].to_numpy()

            res_win = compare_kernels_by_window_size(
                t_full, side_full,
                sizes_min=win_sizes,
                n_per_size=n_windows,
                seed=seed,
                verbose=False
            )

            for w in win_sizes:
                r_b, r_s = res_win["bivariate"][w], res_win["sum_exp"][w]
                for label, r in [("Bivariate", r_b), ("Sum-Exp", r_s)]:
                    n = r['n_valid']
                    f.write(f"{str(w)+' min':<10} | {label:<12} | N valid: {n}\n")
                    if n > 0:
                        for dim, lab in [(1, "Buy"), (2, "Sell")]:
                            c = r['counts'][dim]
                            s = r['stat_mean'][dim]
                            f.write(f"{'':<10} |   {lab:<9} | "
                                    f"KS {100*c['ks']/n:4.0f}% | LB {100*c['lb']/n:4.0f}% | "
                                    f"ED {100*c['er']/n:4.0f}%   "
                                    f"(t moy. KS={s['ks']:.3f} LB={s['lb']:.1f} ED={s['er']:.3f})\n")
                f.write("\n")

            plot_price_volatility(raw_df)

            res_roll_mono = plot_params_across_day(
                t_full, side_full, sizes_min=roll_sizes, kernel='mono',
                min_points=roll_min_points,
            )
            res_roll_sum = plot_params_across_day(
                t_full, side_full, sizes_min=roll_sizes, kernel='sum',
                min_points=roll_min_points,
            )

            f.write("\n--- Rolling params (median on the day) ---\n")
            for name, res_roll in [("Mono-Exp", res_roll_mono), ("Sum-Exp", res_roll_sum)]:
                for w in roll_sizes:
                    d = res_roll[w]
                    if len(d):
                        f.write(f"{name:<9} {w} min : mu_tot~{d['mu_tot'].median():.4f} | "
                                f"eta~{d['eta'].median():.3f} | n_windows={len(d)}\n")
                    else:
                        f.write(f"{name:<9} {w} min : no validated window\n")

    finally:
        plt.show = original_show


if __name__ == "__main__":

    path = "../data/binance_trades_2026-09-01_BTCUSDT.csv.gz"

    df_btc = pd.read_csv(path)

    run_real_data(
        raw_df=df_btc,
        out_dir="btc_results",
        window_start="2026-09-01 13:05:00",   # 1:00 pm (UTC)
        window_end="2026-09-01 13:10:00",     # 1:05 pm (UTC)
        win_sizes=(5, 10, 20, 60),
        n_windows=10,
        roll_sizes=(10,),
    )