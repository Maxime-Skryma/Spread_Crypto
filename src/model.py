import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from scipy.stats import norm

from scipy.stats import kstest, expon
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from tick.hawkes import SimuHawkesExpKernels
from tardis_dev import download_datasets_async


# =============================================================================
# FUNCTIONS
# =============================================================================

# --- Discrete Hawkes Process --------------------------------------------------
def DHP(kernel, gamma, theta, law, n, mu):
    if kernel == "Erlang":
        alpha = np.array([gamma * np.exp(-k * theta) * k for k in range(0, n)])
    if kernel == "Exponential":
        alpha = np.array([gamma * np.exp(-k * theta) for k in range(0, n)])

    alpha[0] = mu
    X_values = np.zeros(n)
    Sn = np.zeros(n)

    if law == "Bernouilli":
        X_values[0] = np.random.binomial(1, alpha[0])
        Sn[0] = X_values[0]
        for k in range(1, n):
            X_values[k] = np.random.binomial(1, alpha[0] + np.sum(alpha[k-1:0:-1] * X_values[1:k]))
            Sn[k] = Sn[k-1] + X_values[k]
    if law == "Poisson":
        X_values[0] = np.random.poisson(alpha[0])
        Sn[0] = X_values[0]
        for k in range(1, n):
            X_values[k] = np.random.poisson(alpha[0] + np.sum(alpha[k-1:0:-1] * X_values[1:k]))
            Sn[k] = Sn[k-1] + X_values[k]

    mu = mu / (1 + alpha[0] - np.sum(alpha))

    if law == "Bernouilli":
        sigma_asy = mu * (1 - mu) / ((1 + alpha[0] - np.sum(alpha))**2)
    if law == "Poisson":
        sigma_asy = mu / ((1 + alpha[0] - np.sum(alpha))**2)

    Sn_div_n = [Sn[k] / (k + 1) for k in range(0, n)]
    sigma_div = np.array([sigma_asy / np.sqrt(k + 1) for k in range(0, n)])

    upper = mu + 1.96 * sigma_div
    lower = mu - 1.96 * sigma_div
    n_values = np.arange(1, n + 1)

    plt.plot(Sn)
    plt.title(f"Discrete Hawkes process with law : {law}, kernel : {kernel}, gamma : {gamma} and theta : {theta}")
    plt.xlabel("Index")
    plt.ylabel("Valeur")
    plt.show()

    plt.plot(Sn_div_n)
    plt.title("Sn/n")
    plt.axhline(mu, color="red")
    plt.fill_between(n_values, lower, upper, color="orange", alpha=0.2, label="95% CI")
    plt.plot(n_values, upper, color="orange", alpha=0.7)
    plt.plot(n_values, lower, color="orange", alpha=0.7)
    plt.xlabel("Index")
    plt.ylabel("Valeur")
    plt.show()


# --- Continuous Hawkes Process (simulation by inverse-CDF) ---------------------
def CHP(mu, beta, gamma, T_max):
    if gamma/beta>=1:
        raise ValueError("Be aware, we need gamma/beta <1 for stationnarity")
    T = 0
    phi = 0
    N = []
    Phi = []

    while True:
        u = np.random.uniform(0, 1)
        F = lambda delta: 1 - (np.exp(-mu * delta - phi * (1 - np.exp(-beta * delta)) / beta))
        b = -np.log(1 - u) / mu + 1e-8
        G_delta = brentq(lambda x: F(x) - u, 0, b)

        T = T + G_delta
        if T > T_max:
            break

        phi = phi * np.exp(-beta * G_delta) + gamma
        N.append(T)
        Phi.append(phi)  # To follow intensity evolution

    return np.array(N), np.array(Phi)

def intensity_path(t_grid, mu, beta, N, Phi):
    idx = np.searchsorted(N, t_grid, side='right') - 1  
    has_event = idx >= 0
    t_last = np.where(has_event, N[np.clip(idx, 0, None)], 0.0)
    E_last = np.where(has_event, Phi[np.clip(idx, 0, None)], 0.0)
    return mu + E_last * np.exp(-beta * (t_grid - t_last))


def CHP_plot(mu, beta, gamma, T_max, n_grid=2000):
    N, Phi = CHP(mu, beta, gamma, T_max)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    title = f"Hawkes Processe and its conditional intensity, with (mu={mu}, beta={beta}, alpha={gamma})"
    fig.suptitle(title, fontsize=14)

    axes[0].step(np.concatenate([[0], N]), np.arange(len(N) + 1), where='post')
    axes[0].set_ylabel('N(t)')
    

    t_grid = np.linspace(0, T_max, n_grid)
    lam = intensity_path(t_grid, mu, beta, N, Phi)
    axes[1].plot(t_grid, lam, lw=1)
    axes[1].axhline(mu, color='red', ls='--', lw=1, label=r'$\mu$')
    axes[1].scatter(N, mu + Phi, color='black', s=8, zorder=3, label=r'$\lambda^*(T_n^+)$')
    axes[1].set_xlabel('t'); axes[1].set_ylabel(r'$\lambda^*(t)$')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# --- Confidence intervals for the continuous process --------------------------
def CHP_IC(v, mu, beta, gamma, T_max):
    mean = (v * mu) / (1 - gamma / beta)
    sigma = np.sqrt(v * mu / (T_max * (1 - gamma / beta)**3))
    return mean, sigma


def CHP_IC_plot(mu, beta, gamma, T_max):
    v = np.linspace(0, 1, 100)
    mean_v = CHP_IC(v, mu, beta, gamma, T_max)[0]
    sigma_v = CHP_IC(v, mu, beta, gamma, T_max)[1]
    lower = mean_v - 1.96 * sigma_v
    upper = mean_v + 1.96 * sigma_v

    for k in range(1, 100):
        N, _ = CHP(mu, beta, gamma, T_max)
        seuil = T_max * v
        N_T_v = np.searchsorted(N, seuil, side='right')
        N_div = N_T_v / T_max
        plt.plot(v, N_div, alpha=0.5)

    plt.title(f"N(Tv)/T (mu={mu}, beta={beta}, gamma={gamma})")
    plt.plot(v, mean_v, color="red")
    plt.fill_between(v, lower, upper, color="orange", alpha=0.2, label="95% CI")
    plt.legend()
    plt.show()


# --- Univariate MLE -----------------------------------------------------------
def neg_log_likelihood(theta, time_stamp):
    mu, R, beta = theta
    gamma = R * beta

    k = len(time_stamp)
    A = np.zeros(k)
    first_sum = np.log(mu + gamma * A[0])

    for i in range(1, k):
        A[i] = (1 + A[i-1]) * np.exp(-beta * (time_stamp[i] - time_stamp[i-1]))
        first_sum += np.log(mu + gamma * A[i])

    last_time = time_stamp[k-1]
    second_sum = np.sum((gamma / beta) * (np.exp(-beta * (last_time - time_stamp)) - 1))

    return -(first_sum - mu * last_time + second_sum)


# --- Univariate residuals -----------------------------------------------------
def hawkes_residuals(theta, time_stamp):
    mu, R, beta = theta
    gamma = R * beta
    k = len(time_stamp)

    u = np.zeros(k - 1)
    A = np.zeros(k)

    for i in range(1, k):
        delta_t = time_stamp[i] - time_stamp[i-1]
        u[i-1] = mu * delta_t + (gamma / beta) * (1 - np.exp(-beta * delta_t)) * (1 + A[i-1])
        A[i] = (1 + A[i-1]) * np.exp(-beta * delta_t)

    return u


# --- Excess-of-dispersion test ------------------------------------------------
def engle_russell_ed_test(u):
    N = len(u)
    sigma2_hat = np.var(u, ddof=1)
    Z_stat = np.sqrt(N) * (sigma2_hat - 1) / np.sqrt(8)
    p_value = 2 * (1 - norm.cdf(np.abs(Z_stat)))
    return sigma2_hat, Z_stat, p_value


# --- Bivariate MLE ------------------------------------------------------------
def neg_log_likelihood_bivariate(theta, df):
    mu1, mu2, R11, R21, R12, R22, beta1, beta2 = theta

    gamma11 = R11 * beta1
    gamma21 = R21 * beta1
    gamma12 = R12 * beta2
    gamma22 = R22 * beta2

    time_stamp = df['time_stamp'].to_numpy()
    side = df['side'].to_numpy()
    k = len(time_stamp)

    S1 = np.zeros(k + 1)
    S2 = np.zeros(k + 1)

    N1 = side.sum()
    N2 = k - N1

    first_sum = np.log(np.where(side[0] == 1, mu1, mu2))

    for i in range(1, k):
        S1[i] = (side[i-1] + S1[i-1]) * np.exp(-beta1 * (time_stamp[i] - time_stamp[i-1]))
        S2[i] = ((1 - side[i-1]) + S2[i-1]) * np.exp(-beta2 * (time_stamp[i] - time_stamp[i-1]))

        if side[i] == 1:
            intensity = mu1 + gamma11 * S1[i] + gamma12 * S2[i]
        else:
            intensity = mu2 + gamma21 * S1[i] + gamma22 * S2[i]
        
        first_sum += np.log(max(intensity, 1e-300))


    last_time = time_stamp[k-1]
    S1[k] = side[k-1] + S1[k-1]
    S2[k] = (1 - side[k-1]) + S2[k-1]

    Re1 = N1 - S1[k]
    Re2 = N2 - S2[k]
    second_sum = (R11 + R21) * Re1 + (R12 + R22) * Re2

    nll = -(first_sum - (mu1 + mu2) * last_time - second_sum)
 
    # safety
    if not np.isfinite(nll):
        return 1e10
 
    return nll

# --- Bivariate constraints ----------------------------------------------------
def spectral_det(theta):
    return (1.0 - theta[2]) * (1.0 - theta[5]) - (theta[4] * theta[3]) - 1e-5


def trace_1(theta):
    return 1.0 - theta[2] - 1e-5


def trace_2(theta):
    return 1.0 - theta[5] - 1e-5


# --- Bivariate residuals ------------------------------------------------------
def hawkes_residuals_bivariate(theta, df):
    mu1, mu2, R11, R21, R12, R22, beta1, beta2 = theta
    time_stamp = df.iloc[:, 0].to_numpy()
    side = df.iloc[:, 1].to_numpy()
    k = len(time_stamp)

    S1 = np.zeros(k)
    S2 = np.zeros(k)
    N1 = np.zeros(k)
    N2 = np.zeros(k)

    Lambda1 = np.zeros(k)
    Lambda2 = np.zeros(k)

    Lambda1[0] = mu1 * time_stamp[0]
    Lambda2[0] = mu2 * time_stamp[0]

    for i in range(1, k):
        delta_t = time_stamp[i] - time_stamp[i-1]
        is_buy = side[i-1]
        is_sell = 1 - side[i-1]

        N1[i] = N1[i-1] + is_buy
        N2[i] = N2[i-1] + is_sell

        S1[i] = (is_buy + S1[i-1]) * np.exp(-beta1 * delta_t)
        S2[i] = (is_sell + S2[i-1]) * np.exp(-beta2 * delta_t)

        Lambda1[i] = mu1 * time_stamp[i] + R11 * (N1[i] - S1[i]) + R12 * (N2[i] - S2[i])
        Lambda2[i] = mu2 * time_stamp[i] + R21 * (N1[i] - S1[i]) + R22 * (N2[i] - S2[i])

    tau_1 = Lambda1[side == 1]
    tau_2 = Lambda2[side == 0]

    u1 = np.diff(tau_1)
    u2 = np.diff(tau_2)

    return u1, u2


# --- Univariate fit  ----------------------------------------------------------
def fit_univariate(time_stamp,
                   theta_init=np.array([0.5, 0.5, 0.3]),
                   bounds=((1e-5, None), (1e-5, 0.999), (1e-5, None)),
                   verbose=True):

    resultat = minimize(
        fun=neg_log_likelihood,
        x0=theta_init,
        args=(time_stamp,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-9}
    )

    if verbose:
        print("\n--- RESULTATS DE L'ESTIMATION (univarie) ---")
        print(f"mu (baseline intensity) estime : {resultat.x[0]:.4f}")
        print(f"Branching Ratio (R) estime     : {resultat.x[1]:.4f}")
        print(f"Beta estime                    : {resultat.x[2]:.4f}")
        print(f"-> Gamma deduit                : {(resultat.x[1] * resultat.x[2]):.4f}")
        print(f"Succes                         : {resultat.success}")

    return resultat


# --- Univariate goodness of fit -----------------------------------------------
def univariate_goodness_of_fit(theta_estime, time_stamp, verbose=True):
    u = hawkes_residuals(theta_estime, time_stamp)

    if verbose:
        print("--- Goodness of fit (univarie) ---")

    ks_stat, ks_pval = kstest(u, 'expon')
    if verbose:
        print(f"KS Test       -> Stat: {ks_stat:.4f} | p-value: {ks_pval:.4f}")

    lb_result = acorr_ljungbox(u, lags=[20], return_df=True)
    lb_pval = lb_result['lb_pvalue'].iloc[0]
    if verbose:
        print(f"Ljung-Box     -> Stat: {lb_result['lb_stat'].iloc[0]:.4f} | p-value: {lb_pval:.4f}")

    sigma2_empirique, ed_stat, ed_pval = engle_russell_ed_test(u)
    if verbose:
        print(f"Variance empirique des residus : {sigma2_empirique:.4f}")
        print(f"Engle-Russell ED Test -> Stat Z: {ed_stat:.4f} | p-value: {ed_pval:.4f}")

    return u, ks_pval, lb_pval, ed_pval


# --- Univariate pass-rate loop  -----------------------------------------------
def pass_rate_univariate(numerous_timestamps=None,
                         n_sim=10,
                         sim_params=dict(mu=1.0, beta=0.3, gamma=0.24, T_max=10000),
                         theta_init=np.array([0.5, 0.5, 0.3]),
                         bounds=((1e-5, None), (1e-5, 0.999), (1e-5, None)),
                         verbose=True):

    count_ks = 0
    count_lb = 0
    count_er = 0
    count_joint = 0

    if numerous_timestamps is None:
        numerous_timestamps = [CHP(**sim_params) for _ in range(n_sim)]

    taille = len(numerous_timestamps)

    for k in range(taille):
        current_timestamps = numerous_timestamps[k]
        resultat = minimize(
            fun=neg_log_likelihood,
            x0=theta_init,
            args=(current_timestamps,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-9}
        )
        theta_estimate = resultat.x
        u = hawkes_residuals(theta_estimate, current_timestamps)

        ks_stat, ks_pval = kstest(u, 'expon')
        lb_result = acorr_ljungbox(u, lags=[20], return_df=True)
        lb_pval = lb_result['lb_pvalue'].iloc[0]
        _, _, er_pval = engle_russell_ed_test(u)

        if ks_pval > 0.05:
            count_ks += 1
        if lb_pval > 0.05:
            count_lb += 1
        if er_pval > 0.05:
            count_er += 1
        if ks_pval > 0.05 and lb_pval > 0.05 and er_pval > 0.05:
            count_joint += 1

    if verbose:
        print(f"\n--- PASS RATES (univarie, sur {taille} simulations) ---")
        print(f"Pass Rate KS : {(count_ks / taille) * 100:.2f}%")
        print(f"Pass Rate LB : {(count_lb / taille) * 100:.2f}%")
        print(f"Pass Rate ED : {(count_er / taille) * 100:.2f}%")
        print(f"Pass Rate Joint (KS n LB n ED) : {(count_joint / taille) * 100:.2f}%")

    return count_ks, count_lb, count_er, count_joint, taille


# --- Bivariate simulation (tick) ----------------------------------------------
def simulate_bivariate(baseline=np.array([0.3, 0.2]),
                       adjacency=np.array([[0.2, 0.1], [0.3, 0.15]]),
                       decays=np.array([[1.0, 1.0], [1.0, 1.0]]),
                       end_time=5000.0,
                       seed=42):
    if not _HAS_TICK:
        raise ImportError("tick n'est pas installe : utilise le simulateur numpy de run_all.py.")

    sim = SimuHawkesExpKernels(
        baseline=baseline,
        adjacency=adjacency,
        decays=decays,
        end_time=end_time,
        seed=seed,
        verbose=False,
    )
    sim.simulate()

    t_buy, t_sell = sim.timestamps[0], sim.timestamps[1]

    df_sim = pd.concat([
        pd.DataFrame({"time_stamp": t_buy,  "side": 1}),
        pd.DataFrame({"time_stamp": t_sell, "side": 0}),
    ], ignore_index=True).sort_values("time_stamp").reset_index(drop=True)

    return df_sim


# --- Bivariate fit ------------------------------------------------------------
def _default_bivariate_bounds():
    return (
        (1e-5, None),  # 0: mu1 > 0
        (1e-5, None),  # 1: mu2 > 0
        (0, None),     # 2: R11 >= 0
        (0, None),     # 3: R21 >= 0
        (0, None),     # 4: R12 >= 0
        (0, None),     # 5: R22 >= 0
        (1e-5, None),  # 6: beta1 > 0
        (1e-5, None)   # 7: beta2 > 0
    )


def fit_bivariate(df,
                  theta_init=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]),
                  method='trust-constr',
                  bounds=None,
                  constraints=None,
                  verbose=True):
    if bounds is None:
        bounds = _default_bivariate_bounds()

    if constraints is None:
        if method == 'trust-constr':
            constraints = [
                NonlinearConstraint(spectral_det, 1e-5, np.inf),
                NonlinearConstraint(trace_1, 1e-5, np.inf),
                NonlinearConstraint(trace_2, 1e-5, np.inf)
            ]
        else:
            constraints = [
                {'type': 'ineq', 'fun': spectral_det},
                {'type': 'ineq', 'fun': trace_1},
                {'type': 'ineq', 'fun': trace_2}
            ]

    if method == 'trust-constr':
        resultat = minimize(
            fun=neg_log_likelihood_bivariate,
            x0=theta_init,
            args=(df,),
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'xtol': 1e-8, 'gtol': 1e-8, 'disp': False}
        )
    else:
        resultat = minimize(
            fun=neg_log_likelihood_bivariate,
            x0=theta_init,
            args=(df,),
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False}
        )

    if verbose:
        print("\n--- Bivariate: Estimated Results ---")
        print(f"Optimization success : {resultat.success}")
        print(f"Status               : {resultat.message}")
        print("\n[Baseline intensity - mu]")
        print(f"mu1 (Dimension 1) : {resultat.x[0]:.6f}")
        print(f"mu2 (Dimension 2) : {resultat.x[1]:.6f}")
        print("\n[Branching ratio matrix - R]")
        print(f"R11 (Self-excitation 1->1) : {resultat.x[2]:.4f}")
        print(f"R21 (Cross-excitation 1->2): {resultat.x[3]:.4f}")
        print(f"R12 (Cross-excitation 2->1): {resultat.x[4]:.4f}")
        print(f"R22 (Self-excitation 2->2) : {resultat.x[5]:.4f}")
        print("\n[beta]")
        print(f"beta1 (Decay / Shocks from 1) : {resultat.x[6]:.4f}")
        print(f"beta2 (Decay / Shocks from 2) : {resultat.x[7]:.4f}")

    return resultat


# --- Bivariate goodness of fit ------------------------------------------------
def bivariate_goodness_of_fit(theta_estimate, df, verbose=True):
    u1, u2 = hawkes_residuals_bivariate(theta_estimate, df)
    residuals = [u1, u2]
    if verbose:
        print("--- Goodness of fit (bivarie) ---")

    results = []
    for u in residuals:
        ks_stat, ks_pval = kstest(u, "expon")
        lb_result = acorr_ljungbox(u, lags=[20], return_df=True)
        lb_pval = lb_result["lb_pvalue"].iloc[0]
        sigma2_empirical, ed_stat, ed_pval = engle_russell_ed_test(u)
        if verbose:
            print(f"KS: {ks_stat:.4f} (p={ks_pval:.4f}) | "
                  f"LB: {lb_result['lb_stat'].iloc[0]:.4f} (p={lb_pval:.4f}) | "
                  f"ED: {ed_stat:.4f} (p={ed_pval:.4f}) | var={sigma2_empirical:.4f}")
        results.append((ks_pval, lb_pval, ed_pval))

    return (u1, u2), results


# --- Bivariate pass-rate loop -------------------------------------------------
def pass_rate_bivariate_sim(numerous_dfs=None,
                            n_sim=10,
                            baseline=np.array([0.3, 0.2]),
                            adjacency=np.array([[0.2, 0.1], [0.3, 0.15]]),
                            decays=np.array([[1.0, 1.0], [1.0, 1.0]]),
                            end_time=5000.0,
                            theta_init=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]),
                            bounds=None,
                            constraints=None,
                            verbose=True):
    if bounds is None:
        bounds = _default_bivariate_bounds()
    if constraints is None:
        constraints = [
            NonlinearConstraint(spectral_det, 1e-5, np.inf),
            NonlinearConstraint(trace_1, 1e-5, np.inf),
            NonlinearConstraint(trace_2, 1e-5, np.inf)
        ]

    count_ks_1, count_ks_2 = 0, 0
    count_lb_1, count_lb_2 = 0, 0
    count_er_1, count_er_2 = 0, 0
    count_joint_1, count_joint_2 = 0, 0
    count_global_joint = 0

    taille = n_sim

    if numerous_dfs is None:
        numerous_dfs = [simulate_bivariate(baseline, adjacency, decays, end_time, seed=i)
                        for i in range(taille)]
    else:
        taille = len(numerous_dfs)

    for k in range(taille):
        current_df = numerous_dfs[k]

        resultat = minimize(
            fun=neg_log_likelihood_bivariate,
            x0=theta_init,
            args=(current_df,),
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'xtol': 1e-8, 'gtol': 1e-8, 'disp': False}
        )

        u1, u2 = hawkes_residuals_bivariate(resultat.x, current_df)

        _, ks_pval_1 = kstest(u1, 'expon')
        lb_pval_1 = acorr_ljungbox(u1, lags=[20], return_df=True)['lb_pvalue'].iloc[0]
        _, _, er_pval_1 = engle_russell_ed_test(u1)
        pass_ks_1 = ks_pval_1 > 0.05
        pass_lb_1 = lb_pval_1 > 0.05
        pass_er_1 = er_pval_1 > 0.05
        if pass_ks_1: count_ks_1 += 1
        if pass_lb_1: count_lb_1 += 1
        if pass_er_1: count_er_1 += 1
        if pass_ks_1 and pass_lb_1 and pass_er_1: count_joint_1 += 1

        _, ks_pval_2 = kstest(u2, 'expon')
        lb_pval_2 = acorr_ljungbox(u2, lags=[20], return_df=True)['lb_pvalue'].iloc[0]
        _, _, er_pval_2 = engle_russell_ed_test(u2)
        pass_ks_2 = ks_pval_2 > 0.05
        pass_lb_2 = lb_pval_2 > 0.05
        pass_er_2 = er_pval_2 > 0.05
        if pass_ks_2: count_ks_2 += 1
        if pass_lb_2: count_lb_2 += 1
        if pass_er_2: count_er_2 += 1
        if pass_ks_2 and pass_lb_2 and pass_er_2: count_joint_2 += 1

        if (pass_ks_1 and pass_lb_1 and pass_er_1) and (pass_ks_2 and pass_lb_2 and pass_er_2):
            count_global_joint += 1

    if verbose:
        print(f"\n--- PASS RATE (bivarie, {taille} simulations) ---")
        print(f"[Dim 1 - Buys ] KS {(count_ks_1/taille)*100:.1f}% | LB {(count_lb_1/taille)*100:.1f}% | ED {(count_er_1/taille)*100:.1f}% | Joint {(count_joint_1/taille)*100:.1f}%")
        print(f"[Dim 2 - Sells] KS {(count_ks_2/taille)*100:.1f}% | LB {(count_lb_2/taille)*100:.1f}% | ED {(count_er_2/taille)*100:.1f}% | Joint {(count_joint_2/taille)*100:.1f}%")
        print(f"[Global 6 tests] {(count_global_joint/taille)*100:.1f}%")

    return {
        "ks_1": count_ks_1, "lb_1": count_lb_1, "er_1": count_er_1, "joint_1": count_joint_1,
        "ks_2": count_ks_2, "lb_2": count_lb_2, "er_2": count_er_2, "joint_2": count_joint_2,
        "global_joint": count_global_joint, "taille": taille,
    }


# --- Bivariate sum-of-exponentials --------------------------------------------
def neg_log_likelihood_bivariate_sum_exp(theta_sum, df):
    mu1, mu2, R11_1, R21_1, R12_1, R22_1, R11_2, R21_2, R12_2, R22_2, \
        beta1_1, beta2_1, beta1_2, beta2_2 = theta_sum

    gamma11_1 = R11_1 * beta1_1
    gamma21_1 = R21_1 * beta1_1
    gamma12_1 = R12_1 * beta2_1
    gamma22_1 = R22_1 * beta2_1
    gamma11_2 = R11_2 * beta1_2
    gamma21_2 = R21_2 * beta1_2
    gamma12_2 = R12_2 * beta2_2
    gamma22_2 = R22_2 * beta2_2

    time_stamp = df['time_stamp'].to_numpy()
    side = df['side'].to_numpy()
    k = len(time_stamp)

    S1_1 = np.zeros(k + 1)
    S2_1 = np.zeros(k + 1)
    S1_2 = np.zeros(k + 1)
    S2_2 = np.zeros(k + 1)

    N1 = side.sum()
    N2 = k - N1

    first_sum = np.log(np.where(side[0] == 1, mu1, mu2))

    for i in range(1, k):
        dt = time_stamp[i] - time_stamp[i-1]
        S1_1[i] = (side[i-1] + S1_1[i-1]) * np.exp(-beta1_1 * dt)
        S2_1[i] = ((1 - side[i-1]) + S2_1[i-1]) * np.exp(-beta2_1 * dt)
        S1_2[i] = (side[i-1] + S1_2[i-1]) * np.exp(-beta1_2 * dt)
        S2_2[i] = ((1 - side[i-1]) + S2_2[i-1]) * np.exp(-beta2_2 * dt)

        if side[i] == 1:
            intensity = mu1 + gamma11_1*S1_1[i] + gamma12_1*S2_1[i] + gamma11_2*S1_2[i] + gamma12_2*S2_2[i]
        else:
            intensity = mu2 + gamma21_1*S1_1[i] + gamma22_1*S2_1[i] + gamma21_2*S1_2[i] + gamma22_2*S2_2[i]

        first_sum += np.log(max(intensity, 1e-300))

    last_time = time_stamp[k-1]
    S1_1[k] = side[k-1] + S1_1[k-1]
    S2_1[k] = (1 - side[k-1]) + S2_1[k-1]
    S1_2[k] = side[k-1] + S1_2[k-1]
    S2_2[k] = (1 - side[k-1]) + S2_2[k-1]

    Re1_1 = N1 - S1_1[k]
    Re2_1 = N2 - S2_1[k]
    Re1_2 = N1 - S1_2[k]
    Re2_2 = N2 - S2_2[k]

    second_sum = (R11_1 + R21_1)*Re1_1 + (R12_1 + R22_1)*Re2_1 + (R11_2 + R21_2)*Re1_2 + (R12_2 + R22_2)*Re2_2

    nll = -(first_sum - (mu1 + mu2)*last_time - second_sum)
    if not np.isfinite(nll):
        return 1e10
    return nll


def fit_bivariate_sum_exp(df, theta_init, bounds_sum=None, verbose=True):
    if bounds_sum is None:
        bounds_sum = (
            (1e-5, None), (1e-5, None),
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            (1e-3, 20.0), (1e-3, 20.0),
            (300, 1500.0), (300, 1500.0)
        )

    def spectral_det_sum(theta):
        r11 = theta[2] + theta[6]
        r21 = theta[3] + theta[7]
        r12 = theta[4] + theta[8]
        r22 = theta[5] + theta[9]
        return (1.0 - r11) * (1.0 - r22) - (r12 * r21) - 1e-5

    A = np.zeros((4, 14))
    A[0, 2] = 1.0;  A[0, 6] = 1.0
    A[1, 5] = 1.0;  A[1, 9] = 1.0
    A[2, 12] = 1.0; A[2, 10] = -1.0
    A[3, 13] = 1.0; A[3, 11] = -1.0
    lb = np.array([-np.inf, -np.inf, 1e-3, 1e-3])
    ub = np.array([1.0 - 1e-5, 1.0 - 1e-5, np.inf, np.inf])

    constraints = [
        LinearConstraint(A, lb, ub),
        NonlinearConstraint(spectral_det_sum, 1e-5, np.inf),
    ]

    resultat = minimize(
        fun=neg_log_likelihood_bivariate_sum_exp,
        x0=theta_init,
        args=(df,),
        method='trust-constr',
        bounds=bounds_sum,
        constraints=constraints,
        options={'maxiter': 2000, 'xtol': 1e-8, 'gtol': 1e-8, 'disp': False}
    )

    if verbose:
        x = resultat.x
        print("\n--- Bivariate Sum-Exp : Estimated results ---")
        print(f"Succes : {resultat.success} | {resultat.message}")
        r11 = x[2] + x[6]; r21 = x[3] + x[7]; r12 = x[4] + x[8]; r22 = x[5] + x[9]
        rho = np.max(np.abs(np.linalg.eigvals(np.array([[r11, r12], [r21, r22]]))))
        print(f"R total: R11={r11:.3f} R21={r21:.3f} R12={r12:.3f} R22={r22:.3f} | rayon spectral={rho:.3f}")

    return resultat


def hawkes_residuals_bivariate_sum_exp(theta_sum, df):
    mu1, mu2, R11_1, R21_1, R12_1, R22_1, R11_2, R21_2, R12_2, R22_2, \
        beta1_1, beta2_1, beta1_2, beta2_2 = theta_sum

    time_stamp = df.iloc[:, 0].to_numpy()
    side = df.iloc[:, 1].to_numpy()
    k = len(time_stamp)

    S1_1 = np.zeros(k); S2_1 = np.zeros(k)
    S1_2 = np.zeros(k); S2_2 = np.zeros(k)
    N1 = np.zeros(k); N2 = np.zeros(k)
    Lambda1 = np.zeros(k); Lambda2 = np.zeros(k)

    Lambda1[0] = mu1 * time_stamp[0]
    Lambda2[0] = mu2 * time_stamp[0]

    for i in range(1, k):
        dt = time_stamp[i] - time_stamp[i-1]
        is_buy = side[i-1]
        is_sell = 1 - side[i-1]

        N1[i] = N1[i-1] + is_buy
        N2[i] = N2[i-1] + is_sell

        S1_1[i] = (is_buy + S1_1[i-1]) * np.exp(-beta1_1 * dt)
        S2_1[i] = (is_sell + S2_1[i-1]) * np.exp(-beta2_1 * dt)
        S1_2[i] = (is_buy + S1_2[i-1]) * np.exp(-beta1_2 * dt)
        S2_2[i] = (is_sell + S2_2[i-1]) * np.exp(-beta2_2 * dt)

        Lambda1[i] = (mu1 * time_stamp[i]
                      + R11_1 * (N1[i] - S1_1[i]) + R12_1 * (N2[i] - S2_1[i])
                      + R11_2 * (N1[i] - S1_2[i]) + R12_2 * (N2[i] - S2_2[i]))
        Lambda2[i] = (mu2 * time_stamp[i]
                      + R21_1 * (N1[i] - S1_1[i]) + R22_1 * (N2[i] - S2_1[i])
                      + R21_2 * (N1[i] - S1_2[i]) + R22_2 * (N2[i] - S2_2[i]))

    tau_1 = Lambda1[side == 1]
    tau_2 = Lambda2[side == 0]
    u1 = np.diff(tau_1)
    u2 = np.diff(tau_2)
    return u1, u2


def bivariate_goodness_of_fit_sum_exp(theta_estimate, df, verbose=True):
    u1, u2 = hawkes_residuals_bivariate_sum_exp(theta_estimate, df)
    residuals = [u1, u2]
    if verbose:
        print("--- Goodness of fit (sum-exp) ---")

    results = []
    for u in residuals:
        ks_stat, ks_pval = kstest(u, 'expon')
        lb_result = acorr_ljungbox(u, lags=[20], return_df=True)
        lb_pval = lb_result['lb_pvalue'].iloc[0]
        sigma2_empirique, ed_stat, ed_pval = engle_russell_ed_test(u)
        if verbose:
            print(f"KS: {ks_stat:.4f} (p={ks_pval:.4f}) | "
                  f"LB: {lb_result['lb_stat'].iloc[0]:.4f} (p={lb_pval:.4f}) | "
                  f"ED: {ed_stat:.4f} (p={ed_pval:.4f}) | var={sigma2_empirique:.4f}")
        results.append((ks_pval, lb_pval, ed_pval))

    return (u1, u2), results


# --- Window evaluation helpers ------------------------------------------------
def _eval_window_bivariate(current_df, bounds, constraints):
    t_w = current_df['time_stamp'].to_numpy()
    side_w = current_df['side'].to_numpy()
    T_w = t_w[-1]
    N1 = side_w.sum(); N2 = len(side_w) - N1
    dt_mean = np.mean(np.diff(t_w))

    theta_start = np.array([
        max((N1 / T_w) * 0.5, 1e-3), max((N2 / T_w) * 0.5, 1e-3),
        0.1, 0.1, 0.1, 0.1,
        1.0 / dt_mean, 1.0 / dt_mean,
    ])

    resultat = minimize(
        fun=neg_log_likelihood_bivariate,
        x0=theta_start,
        args=(current_df,),
        method='trust-constr',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'xtol': 1e-8, 'gtol': 1e-8, 'disp': False}
    )

    u1, u2 = hawkes_residuals_bivariate(resultat.x, current_df)

    out = {}
    for dim, u in [(1, u1), (2, u2)]:
        ks_t, ks_p = kstest(u, 'expon')                             
        lb_df = acorr_ljungbox(u, lags=[20], return_df=True)
        lb_t = lb_df['lb_stat'].iloc[0]                            
        lb_p = lb_df['lb_pvalue'].iloc[0]
        sigma2, er_t, er_p = engle_russell_ed_test(u)               
        out[dim] = {
            'ks_p': ks_p, 'lb_p': lb_p, 'er_p': er_p, 'sigma2': sigma2,
            'ks_t': ks_t, 'lb_t': lb_t, 'er_t': er_t,              
            'pass_ks': ks_p > 0.05, 'pass_lb': lb_p > 0.05, 'pass_er': er_p > 0.05,
        }
    return out


def _eval_window_sum_exp(current_df, bounds, constraints):
    t_w = current_df['time_stamp'].to_numpy()
    side_w = current_df['side'].to_numpy()
    T_w = t_w[-1]
    N1 = side_w.sum(); N2 = len(side_w) - N1
    dt_mean = np.mean(np.diff(t_w))

    theta_start = np.array([
        (N1 / T_w) * 0.3, (N2 / T_w) * 0.3,
        0.15, 0.15, 0.10, 0.15,
        0.05, 0.05, 0.05, 0.05,
        1.0 / (20 * dt_mean), 1.0 / (20 * dt_mean),
        1.0 / dt_mean,        1.0 / dt_mean,
    ])

    resultat = minimize(
        fun=neg_log_likelihood_bivariate_sum_exp,
        x0=theta_start,
        args=(current_df,),
        method='trust-constr',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'xtol': 1e-8, 'gtol': 1e-8, 'disp': False}
    )

    u1, u2 = hawkes_residuals_bivariate_sum_exp(resultat.x, current_df)

    out = {}
    for dim, u in [(1, u1), (2, u2)]:
        ks_t, ks_p = kstest(u, 'expon')
        lb_df = acorr_ljungbox(u, lags=[20], return_df=True)
        lb_t = lb_df['lb_stat'].iloc[0]                             # <-- corrige (etait ['lb_pvalue'].iloc[1])
        lb_p = lb_df['lb_pvalue'].iloc[0]
        sigma2, er_t, er_p = engle_russell_ed_test(u)
        out[dim] = {
            'ks_p': ks_p, 'lb_p': lb_p, 'er_p': er_p, 'sigma2': sigma2,
            'pass_ks': ks_p > 0.05, 'pass_lb': lb_p > 0.05, 'pass_er': er_p > 0.05,
            'ks_t': ks_t, 'lb_t': lb_t, 'er_t': er_t,
        }
    return out


# --- QQ overlay ---------------------------------------------------------------
def qq_overlay(resid_mono, resid_sum, title=""):
    def quantiles(u):
        u_sorted = np.sort(u)
        n = len(u_sorted)
        p = (np.arange(1, n + 1) - 0.5) / n
        return expon.ppf(p), u_sorted

    theo_m, emp_m = quantiles(resid_mono)
    theo_s, emp_s = quantiles(resid_sum)

    plt.figure(figsize=(6, 6))
    plt.scatter(theo_m, emp_m, s=5, alpha=0.4, label="Mono-exp", color="tab:red")
    plt.scatter(theo_s, emp_s, s=5, alpha=0.4, label="Sum-exp",  color="tab:blue")

    lim = max(theo_m.max(), emp_m.max(), theo_s.max(), emp_s.max())
    plt.plot([0, lim], [0, lim], 'k--', label="y = x (fit parfait)")

    plt.xlabel("Theoretical Quantile Exp(1)")
    plt.ylabel("Empirical Quantiles of residuals")
    plt.title(f"QQ-plot : mono-exp vs sum-exp — {title}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# --- Window sampling / aggregation --------------------------------------------
def _sample_random_windows(t, side, duration, n_per_size, rng, min_points,
                           max_attempts_factor=20):
    t = np.asarray(t)
    side = np.asarray(side)
    T_total = t[-1]
    n_valid = 0
    attempts = 0
    max_attempts = n_per_size * max_attempts_factor

    while n_valid < n_per_size and attempts < max_attempts:
        attempts += 1
        start = rng.uniform(0, T_total - duration)
        end = start + duration
        mask = (t >= start) & (t < end)
        if mask.sum() < min_points:
            continue
        t_win = t[mask] - t[mask][0]
        side_win = side[mask]
        n_valid += 1
        yield pd.DataFrame({"time_stamp": t_win, "side": side_win})


def _aggregate_pass_rates(t, side, eval_window_fn, bounds, constraints,
                          sizes_min, n_per_size, seed, min_points):
    rng = np.random.default_rng(seed)
    results = {}

    for size_min in sizes_min:
        duration = size_min * 60
        counts = {1: dict(ks=0, lb=0, er=0, joint=0),
                  2: dict(ks=0, lb=0, er=0, joint=0)}
        count_global = 0
        n_valid = 0
        sigma2_list = {1: [], 2: []}
        stat_list = {1: dict(ks=[], lb=[], er=[]),                  # <-- AJOUT
                     2: dict(ks=[], lb=[], er=[])}

        windows = _sample_random_windows(t, side, duration, n_per_size, rng, min_points)

        for df_win in windows:
            try:
                res = eval_window_fn(df_win, bounds, constraints)
            except Exception:
                continue
            n_valid += 1
            for dim in (1, 2):
                r = res[dim]
                if r["pass_ks"]: counts[dim]["ks"] += 1
                if r["pass_lb"]: counts[dim]["lb"] += 1
                if r["pass_er"]: counts[dim]["er"] += 1
                if r["pass_ks"] and r["pass_lb"] and r["pass_er"]:
                    counts[dim]["joint"] += 1
                sigma2_list[dim].append(r["sigma2"])
                stat_list[dim]["ks"].append(r["ks_t"])              # <-- AJOUT
                stat_list[dim]["lb"].append(r["lb_t"])              # <-- AJOUT
                stat_list[dim]["er"].append(r["er_t"])              # <-- AJOUT
            if (res[1]["pass_ks"] and res[1]["pass_lb"]) and (res[1]["pass_ks"] and res[1]["pass_er"]) and (res[1]["pass_lb"] and res[1]["pass_er"]):
                count_global += 1

        results[size_min] = {
            "n_valid": n_valid,
            "counts": counts,
            "global": count_global,
            "sigma2_median": {d: (np.median(sigma2_list[d]) if sigma2_list[d] else np.nan)
                              for d in (1, 2)},
            "stat_mean": {d: {test: (np.mean(stat_list[d][test]) if stat_list[d][test] else np.nan)  # <-- AJOUT
                              for test in ("ks", "lb", "er")}
                          for d in (1, 2)},
        }

    return results


def _print_pass_rate_summary(results, sizes_min):
    for size_min in sizes_min:
        r = results[size_min]
        n = r["n_valid"]
        if n == 0:
            print(f"\n=== {size_min} min: no valid window ===")
            continue
        print(f"\n=== {size_min}-min windows  ({n} windows) ===")
        for dim, label in [(1, "Buys"), (2, "Sells")]:
            c = r["counts"][dim]
            s = r["stat_mean"][dim]                                 # <-- AJOUT
            print(f"  [{label}]  KS {100*c['ks']/n:.0f}% | "
                  f"LB {100*c['lb']/n:.0f}% | ED {100*c['er']/n:.0f}% | "
                  f"Joint {100*c['joint']/n:.0f}%  "
                  f"(median residual variance: {r['sigma2_median'][dim]:.3f})")
            print(f"           stats moy.: KS={s['ks']:.3f} | LB={s['lb']:.1f} | ED={s['er']:.3f}")  # <-- AJOUT


def pass_rate_by_window_size_bivariate(t, side, sizes_min=(5, 10, 20),
                                       n_per_size=10, seed=42, min_points=30,
                                       verbose=True):
    bounds = _default_bivariate_bounds()
    constraints = [
        NonlinearConstraint(spectral_det, 1e-5, np.inf),
        NonlinearConstraint(trace_1, 1e-5, np.inf),
        NonlinearConstraint(trace_2, 1e-5, np.inf),
    ]
    results = _aggregate_pass_rates(
        t, side, _eval_window_bivariate, bounds, constraints,
        sizes_min, n_per_size, seed, min_points,
    )
    if verbose:
        _print_pass_rate_summary(results, sizes_min)
    return results


def pass_rate_by_window_size_sum_exp(t, side, sizes_min=(5, 10, 20),
                                     n_per_size=10, seed=42, min_points=50,
                                     verbose=True):
    bounds = (
        (1e-5, None), (1e-5, None),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (1e-3, 1000.0), (1e-3, 1000.0), (1e-3, 1000.0), (1e-3, 1000.0),
    )

    def spectral_det_sum(theta):
        r11 = theta[2] + theta[6]; r21 = theta[3] + theta[7]
        r12 = theta[4] + theta[8]; r22 = theta[5] + theta[9]
        return (1.0 - r11) * (1.0 - r22) - (r12 * r21) - 1e-5

    A = np.zeros((4, 14))
    A[0, 2] = 1.0;  A[0, 6] = 1.0
    A[1, 5] = 1.0;  A[1, 9] = 1.0
    A[2, 12] = 1.0; A[2, 10] = -1.0
    A[3, 13] = 1.0; A[3, 11] = -1.0
    lb = np.array([-np.inf, -np.inf, 1e-3, 1e-3])
    ub = np.array([1.0 - 1e-5, 1.0 - 1e-5, np.inf, np.inf])
    constraints = [
        LinearConstraint(A, lb, ub),
        NonlinearConstraint(spectral_det_sum, 1e-5, np.inf),
    ]

    results = _aggregate_pass_rates(
        t, side, _eval_window_sum_exp, bounds, constraints,
        sizes_min, n_per_size, seed, min_points,
    )
    if verbose:
        _print_pass_rate_summary(results, sizes_min)
    return results


def compare_kernels_by_window_size(t, side, sizes_min=(5, 10, 20),
                                   n_per_size=10, seed=42, verbose=True):
    results_biv = pass_rate_by_window_size_bivariate(
        t, side, sizes_min=sizes_min, n_per_size=n_per_size, seed=seed, verbose=False,
    )
    results_sum = pass_rate_by_window_size_sum_exp(
        t, side, sizes_min=sizes_min, n_per_size=n_per_size, seed=seed, verbose=False,
    )
    if verbose:
        print(f"{'Window':<10} {'Model':<12} {'Global pass rate':<18} {'n valid'}")
        print("-" * 55)
        for size_min in sizes_min:
            for label, res in [("Bivariate", results_biv), ("Sum-exp", results_sum)]:
                r = res[size_min]
                n = r["n_valid"]
                rate = f"{100 * r['global'] / n:.1f}%" if n > 0 else "n/a"
                print(f"{size_min} min{'':<4} {label:<12} {rate:<18} {n}")
            print()
    return {"bivariate": results_biv, "sum_exp": results_sum}

def _fit_window_sum_exp(current_df, bounds, constraints):
    t_w   = current_df['time_stamp'].to_numpy()
    side_w = current_df['side'].to_numpy()
    T_w = t_w[-1]
    N1 = side_w.sum(); N2 = len(side_w) - N1
    dt_mean = np.mean(np.diff(t_w))

    theta_start = np.array([
        (N1 / T_w) * 0.3, (N2 / T_w) * 0.3,
        0.15, 0.15, 0.10, 0.15,
        0.05, 0.05, 0.05, 0.05,
        1.0 / (20 * dt_mean), 1.0 / (20 * dt_mean),
        1.0 / dt_mean,        1.0 / dt_mean,
    ])
    res = minimize(
        fun=neg_log_likelihood_bivariate_sum_exp, x0=theta_start, args=(current_df,),
        method='trust-constr', bounds=bounds, constraints=constraints,
        options={'maxiter': 2000, 'xtol': 1e-8, 'gtol': 1e-8, 'disp': False},
    )
    return res.x, res.success


def estimate_params_rolling(t, side, size_min, kernel='mono',
                            step_min=None, min_points=30):
    t = np.asarray(t); side = np.asarray(side)
    duration = size_min * 60.0
    step = duration if step_min is None else step_min * 60.0

    if kernel == 'mono':
        fit_fn = _fit_window_bivariate
        bounds = _default_bivariate_bounds()
        constraints = [
            NonlinearConstraint(spectral_det, 1e-5, np.inf),
            NonlinearConstraint(trace_1, 1e-5, np.inf),
            NonlinearConstraint(trace_2, 1e-5, np.inf),
        ]
        def branching(theta):
            return np.array([[theta[2], theta[4]],
                             [theta[3], theta[5]]])
    elif kernel == 'sum':
        fit_fn = _fit_window_sum_exp
        bounds = (
            (1e-5, None), (1e-5, None),
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            (1e-3, 1000.0), (1e-3, 1000.0), (1e-3, 1000.0), (1e-3, 1000.0),
        )
        def spectral_det_sum(theta):
            r11 = theta[2] + theta[6]; r21 = theta[3] + theta[7]
            r12 = theta[4] + theta[8]; r22 = theta[5] + theta[9]
            return (1.0 - r11) * (1.0 - r22) - (r12 * r21) - 1e-5
        A = np.zeros((4, 14))
        A[0, 2] = 1.0;  A[0, 6] = 1.0
        A[1, 5] = 1.0;  A[1, 9] = 1.0
        A[2, 12] = 1.0; A[2, 10] = -1.0
        A[3, 13] = 1.0; A[3, 11] = -1.0
        lb = np.array([-np.inf, -np.inf, 1e-3, 1e-3])
        ub = np.array([1.0 - 1e-5, 1.0 - 1e-5, np.inf, np.inf])
        constraints = [LinearConstraint(A, lb, ub),
                       NonlinearConstraint(spectral_det_sum, 1e-5, np.inf)]
        def branching(theta):
            return np.array([[theta[2] + theta[6], theta[4] + theta[8]],   # R total = echelle1 + echelle2
                             [theta[3] + theta[7], theta[5] + theta[9]]])
    else:
        raise ValueError("kernel doit etre 'mono' ou 'sum'")

    T_total = t[-1]
    rows = []
    start = 0.0
    while start + duration <= T_total:
        end = start + duration
        mask = (t >= start) & (t < end)
        if mask.sum() >= min_points:
            df_win = pd.DataFrame({'time_stamp': t[mask] - t[mask][0], 'side': side[mask]})
            try:
                theta, ok = fit_fn(df_win, bounds, constraints)
                eta = np.max(np.abs(np.linalg.eigvals(branching(theta))))
                rows.append({'center_h': (start + duration/2) / 3600.0,
                             'mu1': theta[0], 'mu2': theta[1], 'mu_tot': theta[0] + theta[1],
                             'eta': eta, 'n': int(mask.sum()), 'ok': ok})
            except Exception:
                pass
        start += step
    return pd.DataFrame(rows)


def plot_params_across_day(t, side, sizes_min=(10, 30, 60), kernel='mono', min_points=30):
    colors = {10: 'green', 30: 'red', 60: 'orange'}
    res = {s: estimate_params_rolling(t, side, s, kernel=kernel, min_points=min_points)
           for s in sizes_min}

    fig, (ax_mu, ax_eta) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for s in sizes_min:
        d = res[s]
        if len(d) == 0:
            continue
        ax_mu.plot(d['center_h'],  d['mu_tot'], '.', ms=4, color=colors.get(s), label=f'{s} min')
        ax_eta.plot(d['center_h'], d['eta'],    '.', ms=4, color=colors.get(s), label=f'{s} min')

    kern = 'mono-exp' if kernel == 'mono' else 'sum-exp'
    ax_mu.set_ylabel(r'Baseline $\mu_1+\mu_2$ (evts/s)')
    ax_mu.set_title(f'Parametres en fenetres glissantes ({kern})')
    ax_mu.legend(); ax_mu.grid(alpha=0.3)
    ax_eta.axhline(1.0, color='k', ls='--', lw=0.8)
    ax_eta.set_ylim(0, 1.05); ax_eta.set_xlabel('heure dans la serie (h)')
    ax_eta.set_ylabel(r'Branching ratio $\eta = \rho(R)$')
    ax_eta.legend(); ax_eta.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    return res

