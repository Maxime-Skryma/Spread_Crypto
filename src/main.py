import os
import matplotlib.pyplot as plt
import pandas as pd

from model import (
    fit_bivariate, bivariate_goodness_of_fit, 
    fit_bivariate_sum_exp, bivariate_goodness_of_fit_sum_exp,
    compare_kernels_by_window_size, qq_overlay
)
from data import (
    build_hawkes_dataframe, initialize_hawkes_params,
    initialize_hawkes_params_sum_exp
)

def run_real_data(
    raw_df,
    out_dir="results",
    win_sizes=(5, 10, 20),
    n_windows=10,
    seed=42
):
    os.makedirs(out_dir, exist_ok=True)
    
    counter = [1]
    original_show = plt.show
    def save_and_close(*args, **kwargs):
        plt.savefig(f"{out_dir}/fig_{counter[0]:02d}.png", bbox_inches='tight', dpi=300)
        counter[0] += 1
        plt.close()
    plt.show = save_and_close

    with open(f"{out_dir}/results_summary.txt", "w") as f:
        f.write("=== REAL DATA RESULTS ===\n\n")
        
        # Preprocessing
        df_hawkes = build_hawkes_dataframe(raw_df)
        t_real = df_hawkes['time_stamp'].to_numpy()
        side_real = df_hawkes['side'].to_numpy()

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

        # 3. Fit BivariateGlobal (Sum-Exp) and figures
        theta_init_sum = initialize_hawkes_params_sum_exp(t_real, side_real)
        res_sum = fit_bivariate_sum_exp(
            df_hawkes, 
            theta_init=theta_init_sum,
            verbose=False
        )
        
        (u1_sum, u2_sum), gof_sum = bivariate_goodness_of_fit_sum_exp(res_sum.x, df_hawkes, verbose=False)

        f.write("--- Global Bivariate Fit (Sum-Exp) ---\n")
        f.write(f"Dim 1 (Buy)  -> KS: {gof_sum[0][0]:.4f} | LB: {gof_sum[0][1]:.4f} | ED: {gof_sum[0][2]:.4f}\n")
        f.write(f"Dim 2 (Sell) -> KS: {gof_sum[1][0]:.4f} | LB: {gof_sum[1][1]:.4f} | ED: {gof_sum[1][2]:.4f}\n\n")

        # PNG
        qq_overlay(u1_mono, u1_sum, title="Dimension 1 (Buys)")
        qq_overlay(u2_mono, u2_sum, title="Dimension 2 (Sells)")

        # Windows
        f.write("--- Window Comparison ---\n")
        f.write(f"{'Window':<10} | {'Model':<12} | {'Pass Rate':<10} | {'N Valid'}\n")
        f.write("-" * 50 + "\n")
        
        res_win = compare_kernels_by_window_size(
            t_real, side_real, 
            sizes_min=win_sizes, 
            n_per_size=n_windows, 
            seed=seed, 
            verbose=False
        )
        
        for w in win_sizes:
            r_b, r_s = res_win["bivariate"][w], res_win["sum_exp"][w]
            p_b = f"{100*r_b['global']/r_b['n_valid']:.1f}%" if r_b['n_valid'] > 0 else "N/A"
            p_s = f"{100*r_s['global']/r_s['n_valid']:.1f}%" if r_s['n_valid'] > 0 else "N/A"
            f.write(f"{str(w)+' min':<10} | {'Bivariate':<12} | {p_b:<10} | {r_b['n_valid']}\n")
            f.write(f"{str(w)+' min':<10} | {'Sum-Exp':<12} | {p_s:<10} | {r_s['n_valid']}\n")

    plt.show = original_show

if __name__ == "__main__":
    
    chemin = "../data/binance_trades_2026-09-01_BTCUSDT.csv.gz"
    
    df_btc = pd.read_csv(chemin)
    
    run_real_data(
        raw_df=df_btc, 
        out_dir="btc_results", 
        win_sizes=(5, 10, 20), 
        n_windows=10
    )