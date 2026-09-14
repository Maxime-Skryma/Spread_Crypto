import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import norm

from scipy.stats import kstest, expon
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from tick.hawkes import SimuHawkesExpKernels


# =============================================================================
# FUNCTIONS
# =============================================================================

# --- Discrete Hawkes Process --------------------------------------------------
# Let's first try to modelise a Discrete Hawkes process
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

    mu = mu / (1 + alpha[0] - np.sum(alpha))  # on veut pas d'mu or np.sum(alpha) l'inclut

    if law == "Bernouilli":
        sigma_asy = mu * (1 - mu) / ((1 + alpha[0] - np.sum(alpha))**2)  # same
    if law == "Poisson":
        sigma_asy = mu / ((1 + alpha[0] - np.sum(alpha))**2)  # same

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
    plt.fill_between(
        n_values,
        lower,
        upper,
        color="orange",
        alpha=0.2,
        label="95% CI")

    plt.plot(n_values, upper, color="orange", alpha=0.7)
    plt.plot(n_values, lower, color="orange", alpha=0.7)
    plt.xlabel("Index")
    plt.ylabel("Valeur")
    plt.show()


# --- Continuous Hawkes Process (simulation by inverse-CDF) ---------------------
# Let's try to code a continuous Hawkes Process
# First let's try to just inverse the distribution function
# Then we will use the Thinning d'Ogatta
#
# We can define it so easily and recursively just because we have an exponential
# kernel, which allows to just multiply each time.
# For calculation, I'll add a note on how we came to such an easy loop !
def CHP(mu, beta, gamma, T_max):

    if gamma / beta >= 1:
        raise ValueError("Attention, il est nécessaire d'avoir : γ/β < 1 afin d'assurer la condition sur la fonction d'excitation")

    # alpha 0 : intensité de fond (baseline)
    # gamma : amplitude du noyau d'excitation
    # beta : taux de décroissance

    T = 0
    phi = 0

    N = []

    while True:
        u = np.random.uniform(0, 1)
        F = lambda delta: 1 - (np.exp(-mu * delta - phi * (1 - np.exp(-beta * delta)) / beta))  # Fonction de répartition de Δn+1 appliqué en δ (on part de la fonction de survie)

        b = -np.log(1 - u) / mu + 1e-8   # borne haute garantie (car S(b) >= 1-e^{-mu*b} = u) : résultat analytique (see the note)
        # On utilise cette borne haute pour être sûr que S(b)>0 et donc que l'algo brentq (qui nécessite au moins une annulation sur l'intervalle fonctionne)

        G_delta = brentq(lambda x: F(x) - u, 0, b)  # G_delta est une simulation de la variable aléatoire Δn+1

        T = T + G_delta  # Tn+1 = Tn + Δn+1   (ici G_delta car on utilise une simulation)
        if T > T_max:
            break  # Permet de sortir de la boucle dès que le T calculé dépasse le T_max fixé

        phi = phi * np.exp(-beta * G_delta) + gamma  # Φn+1 = Φn * exp(-β * Δn+1) + γ

        N.append(T)  # N is totally encoded by (T1,...,Tn)

    N = np.array(N)

    return N


def CHP_plot(mu, beta, gamma, T_max):
    N = CHP(mu, beta, gamma, T_max)
    plt.step(np.concatenate([[0], N]), np.arange(len(N) + 1), where='post')
    plt.xlabel('t')
    plt.ylabel('N(t)')
    plt.title('Processus de Hawkes continu')
    plt.grid(True, alpha=0.3)
    plt.show()


# --- Confidence intervals for the continuous process --------------------------
# Confidence intervals which depend on v
# We implement the analytic "solution"

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

    # Pour la boucle : ATTENTION, ça ne veut pas dire que 95% seront entièrement dans l'intervalle de confiance
    # Notre intervalle de confiance fonctionne à v fixé, ça serait plutôt donc :
    # à v fixé, 95% des 100 points (évaluation de la trajectoire en v) se situe dans l'intervalle de confiance

    # Pour pouvoir déduire que 95% des trajectoires sont censées être situées dans notre intervalle, il faudrait qu'on ai un intervalle indépendant de v
    # Ce qui est la prochaine étape

    for k in range(1, 100):  # Pour afficher une multitude de simulations de trajectoires (indépendantes)
        N = CHP(mu, beta, gamma, T_max)

        seuil = T_max * v
        N_T_v = np.searchsorted(N, seuil, side='right')

        N_div = N_T_v / T_max  # = N(Tv)/T : c'est pour celui-ci qu'on veut faire un intervalle de confiance
        plt.plot(v, N_div, alpha=0.5)

    plt.title("N(Tv)/T")
    plt.plot(v, mean_v, color="red")
    plt.fill_between(
        v,
        lower,
        upper,
        color="orange",
        alpha=0.2,
        label="95% CI")
    plt.legend()
    plt.show()


# --- Univariate MLE -----------------------------------------------------------
# The complexity should be O(len(time_stamp)) thanks to recursion
def neg_log_likelihood(theta, time_stamp):

    mu, R, beta = theta  # Where R is the branchement ratio (which should be 0<.<1)
    # We use the branchement ratio, because it allows to use numerical methods more robusts to constraints

    gamma = R * beta  # we recalculate gamma

    k = len(time_stamp)
    A = np.zeros(k)

    first_sum = np.log(mu + gamma * A[0])

    # First sum
    for i in range(1, k):
        A[i] = (1 + A[i-1]) * np.exp(-beta * (time_stamp[i] - time_stamp[i-1]))
        first_sum += np.log(mu + gamma * A[i])

    # Second sum

    last_time = time_stamp[k-1]

    second_sum = np.sum((gamma / beta) * (np.exp(-beta * (last_time - time_stamp)) - 1))  # time_stamp is a vector, so everything inside np.sum is a vector

    return -(first_sum - mu * last_time + second_sum)


# --- Univariate residuals -----------------------------------------------------
def hawkes_residuals(theta, time_stamp):
    mu, R, beta = theta
    gamma = R * beta
    k = len(time_stamp)

    u = np.zeros(k - 1)
    A = np.zeros(k)

    # we apply the change theorem
    # We recursively calculate residuals : proof on paper
    for i in range(1, k):
        delta_t = time_stamp[i] - time_stamp[i-1]

        u[i-1] = mu * delta_t + (gamma / beta) * (1 - np.exp(-beta * delta_t)) * (1 + A[i-1])

        # Update
        A[i] = (1 + A[i-1]) * np.exp(-beta * delta_t)

    return u


# --- Excess-of-dispersion test ------------------------------------------------
# Test of excess of dispersion
# H0 : the variance of residuals is 1
def engle_russell_ed_test(u):

    N = len(u)

    # calculation of empirical dispersion (ddof = degrees of liberty = 1 to be unbiased)
    sigma2_hat = np.var(u, ddof=1)

    # We calculate our test statistic (that follows N(0,1) when N-> +inf)
    Z_stat = np.sqrt(N) * (sigma2_hat - 1) / np.sqrt(8)

    # bilateral p_value
    p_value = 2 * (1 - norm.cdf(np.abs(Z_stat)))

    return sigma2_hat, Z_stat, p_value


# --- Bivariate MLE ------------------------------------------------------------
def neg_log_likelihood_bivariate(theta, df):

    mu1, mu2, R11, R21, R12, R22, beta1, beta2 = theta  # We use branchment ratio for numerical optimisation

    # We deduce gamma using branchment ratio
    gamma11 = R11 * beta1
    gamma21 = R21 * beta1
    gamma12 = R12 * beta2
    gamma22 = R22 * beta2

    # time_stamp and order type
    time_stamp = df['time_stamp'].to_numpy()
    side = df['side'].to_numpy()

    k = len(time_stamp)

    S1 = np.zeros(k + 1)
    S2 = np.zeros(k + 1)

    N1 = side.sum()
    N2 = k - N1

    first_sum = np.log(np.where(side[0] == 1, mu1, mu2))

    # First sum
    for i in range(1, k):
        S1[i] = (side[i-1] + S1[i-1]) * np.exp(-beta1 * (time_stamp[i] - time_stamp[i-1]))

        S2[i] = ((1 - side[i-1]) + S2[i-1]) * np.exp(-beta2 * (time_stamp[i] - time_stamp[i-1]))

        if side[i] == 1:
            first_sum += np.log(mu1 + gamma11 * S1[i] + gamma12 * S2[i])
        else:
            first_sum += np.log(mu2 + gamma21 * S1[i] + gamma22 * S2[i])

    # Second sum

    last_time = time_stamp[k-1]

    S1[k] = side[k-1] + S1[k-1]

    S2[k] = (1 - side[k-1]) + S2[k-1]

    Re1 = N1 - S1[k]
    Re2 = N2 - S2[k]

    second_sum = (R11 + R21) * Re1 + (R12 + R22) * Re2

    return -(first_sum - (mu1 + mu2) * last_time - second_sum)


# --- Bivariate constraints ----------------------------------------------------
# Numerous constraints to respect :
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

    # The compensator at the first event, only depends on passed time
    Lambda1[0] = mu1 * time_stamp[0]
    Lambda2[0] = mu2 * time_stamp[0]

    # Evaluation of the global compensator at each instant t_i
    for i in range(1, k):
        delta_t = time_stamp[i] - time_stamp[i-1]

        is_buy = side[i-1]
        is_sell = 1 - side[i-1]

        # update of cumulative counts N(t)
        N1[i] = N1[i-1] + is_buy
        N2[i] = N2[i-1] + is_sell

        # update of historical chocs S(t)
        S1[i] = (is_buy + S1[i-1]) * np.exp(-beta1 * delta_t)
        S2[i] = (is_sell + S2[i-1]) * np.exp(-beta2 * delta_t)

        # We use the exact (analytical) primitive
        Lambda1[i] = mu1 * time_stamp[i] + R11 * (N1[i] - S1[i]) + R12 * (N2[i] - S2[i])
        Lambda2[i] = mu2 * time_stamp[i] + R21 * (N1[i] - S1[i]) + R22 * (N2[i] - S2[i])

    # We use change time theorem
    # We separate according to the side
    tau_1 = Lambda1[side == 1]
    tau_2 = Lambda2[side == 0]

    # the residuals u_i are the time between events in the new reper
    u1 = np.diff(tau_1)
    u2 = np.diff(tau_2)

    return u1, u2




# --- Univariate fit  -------------------------------------------------
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
        print("\n--- RÉSULTATS DE L'ESTIMATION ---")
        print(f"mu (baseline intensity) estimé      : {resultat.x[0]:.4f}")
        print(f" Branchement Ratio (R) estimé   : {resultat.x[1]:.4f}")
        print(f"Beta estimé        : {resultat.x[2]:.4f}")
        print(f"-> Gamma déduit    : {(resultat.x[1] * resultat.x[2]):.4f}")
        print(f"Succès             : {resultat.success}")

    return resultat


# --- Univariate goodness of fit (cells 17-18) ---------------------------------
def univariate_goodness_of_fit(theta_estime, time_stamp, verbose=True):

    # residuals estimated
    u = hawkes_residuals(theta_estime, time_stamp)

    if verbose:
        print("--- Goodness of fit ---")

    # TEST 1 : Kolmogorov-Smirnov (Marginal distribution Exp(1))
    # H0 : data follows an Exp(1) distribution
    ks_stat, ks_pval = kstest(u, 'expon')
    if verbose:
        print(f"KS Test       -> Stat: {ks_stat:.4f} | p-value: {ks_pval:.4f}")

    # TEST 2 : Ljung-Box (Linear AC until the 20th lag)
    # H0 : residuals are independant (no autocorrelation)
    lb_result = acorr_ljungbox(u, lags=[20], return_df=True)
    lb_pval = lb_result['lb_pvalue'].iloc[0]
    if verbose:
        print(f"Ljung-Box     -> Stat: {lb_result['lb_stat'].iloc[0]:.4f} | p-value: {lb_pval:.4f}")

    sigma2_empirique, ed_stat, ed_pval = engle_russell_ed_test(u)

    if verbose:
        print(f"empirical variance of résidus : {sigma2_empirique:.4f}")
        print(f"Engle-Russell ED Test -> Stat Z: {ed_stat:.4f} | p-value: {ed_pval:.4f}")

    return u, ks_pval, lb_pval, ed_pval


# --- Univariate pass-rate loop (cell 19) --------------------------------------
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

    # Let's try to implement the pass_rate idea of the article
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
        print("\n--- RÉSULTATS DES PASS RATES (sur 50 simulations) ---")
        print(f"Pass Rate KS : {(count_ks / taille) * 100:.2f}%")
        print(f"Pass Rate LB : {(count_lb / taille) * 100:.2f}%")
        print(f"Pass Rate ED : {(count_er / taille) * 100:.2f}%")
        print(f"Pass Rate Joint (KS ∩ LB ∩ ED) : {(count_joint / taille) * 100:.2f}%")

    return count_ks, count_lb, count_er, count_joint, taille


# --- Bivariate simulation (cell 20) -------------------------------------------
def simulate_bivariate(baseline=np.array([0.3, 0.2]),
                       adjacency=np.array([[0.2, 0.1], [0.3, 0.15]]),
                       decays=np.array([[1.0, 1.0], [1.0, 1.0]]),
                       end_time=5000.0,
                       seed=42):

    # --- Simulateur ---
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


# --- Bivariate fit ----------------------------------------------
def _default_bivariate_bounds():
    # Linear constraints
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
    """
    method='trust-constr' -> uses NonlinearConstraint (cell 24)
    method='SLSQP'        -> uses dict 'ineq' constraints (cell 25)
    Pass your own `constraints` to override the method's default.
    """

    if bounds is None:
        bounds = _default_bivariate_bounds()

    if constraints is None:
        if method == 'trust-constr':
            # inequality constraints
            constraints = [
                NonlinearConstraint(spectral_det, 1e-5, np.inf),
                NonlinearConstraint(trace_1, 1e-5, np.inf),
                NonlinearConstraint(trace_2, 1e-5, np.inf)
            ]
        else:
            # Inequalities constraints
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
            method='trust-constr',  # We try another algorithm : which is more robust to
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'xtol': 1e-8, 'gtol': 1e-8, 'disp': True}
        )
    else:
        resultat = minimize(
            fun=neg_log_likelihood_bivariate,
            x0=theta_init,
            args=(df,),
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-9, 'disp': True}
        )

    if verbose:
        print("\n--- Bivariate : Estimated results ---")
        print(f"Succès de l'optimisation : {resultat.success}")
        print(f"Statut                   : {resultat.message}")

        print("\n[Baseline intensity - mu]")
        print(f"mu1 (Dimension 1) : {resultat.x[0]:.6f}")
        print(f"mu2 (Dimension 2) : {resultat.x[1]:.6f}")

        print("\n[Branchement ratio matrix - R]")
        print(f"R11 (Auto-excitation 1->1)   : {resultat.x[2]:.4f}")
        print(f"R21 (Croisée 1->2)           : {resultat.x[3]:.4f}")
        print(f"R12 (Croisée 2->1)           : {resultat.x[4]:.4f}")
        print(f"R22 (Auto-excitation 2->2)   : {resultat.x[5]:.4f}")

        print("\n[beta]")
        print(f"beta1 (Chocs issus de 1) : {resultat.x[6]:.4f}")
        print(f"beta2 (Chocs issus de 2) : {resultat.x[7]:.4f}")

        print("\n[gamma]")
        print(f"-> gamma11 : {(resultat.x[2] * resultat.x[6]):.4f}")
        print(f"-> gamma21 : {(resultat.x[3] * resultat.x[6]):.4f}")
        print(f"-> gamma12 : {(resultat.x[4] * resultat.x[7]):.4f}")
        print(f"-> gamma22 : {(resultat.x[5] * resultat.x[7]):.4f}")

    return resultat


# --- Bivariate goodness of fit ----------------------------------
def bivariate_goodness_of_fit(theta_estimate, df, verbose=True):

    # residuals estimated
    u1, u2 = hawkes_residuals_bivariate(theta_estimate, df)

    residuals = [u1, u2]
    if verbose:
        print("--- Goodness of fit ---")

    results = []
    for u in residuals:
        # TEST 1 : Kolmogorov-Smirnov (Marginal distribution Exp(1))
        # H0 : data follows an Exp(1) distribution
        ks_stat, ks_pval = kstest(u, 'expon')
        if verbose:
            print(f"KS Test       -> Stat: {ks_stat:.4f} | p-value: {ks_pval:.4f}")

        # TEST 2 : Ljung-Box (Linear AC until the 20th lag)
        # H0 : residuals are independant (no autocorrelation)
        lb_result = acorr_ljungbox(u, lags=[20], return_df=True)
        lb_pval = lb_result['lb_pvalue'].iloc[0]
        if verbose:
            print(f"Ljung-Box     -> Stat: {lb_result['lb_stat'].iloc[0]:.4f} | p-value: {lb_pval:.4f}")

        sigma2_empirique, ed_stat, ed_pval = engle_russell_ed_test(u)

        if verbose:
            print(f"empirical variance of résidus : {sigma2_empirique:.4f}")
            print(f"Engle-Russell ED Test -> Stat Z: {ed_stat:.4f} | p-value: {ed_pval:.4f}")

        results.append((ks_pval, lb_pval, ed_pval))

    return (u1, u2), results


# --- Bivariate pass-rate loop ---------------------------------------
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

    # All our counters
    count_ks_1, count_ks_2 = 0, 0
    count_lb_1, count_lb_2 = 0, 0
    count_er_1, count_er_2 = 0, 0
    count_joint_1, count_joint_2 = 0, 0
    count_global_joint = 0

    taille = n_sim  # Number of simulations we want

    # Generation of bivariate simulations (using tick)
    if numerous_dfs is None:
        numerous_dfs = []
        for i in range(taille):
            sim = SimuHawkesExpKernels(
                baseline=baseline,
                adjacency=adjacency,
                decays=decays,
                end_time=end_time,
                seed=i,
                verbose=False,
            )

            sim.simulate()
            t_buy, t_sell = sim.timestamps[0], sim.timestamps[1]

            df_sim = pd.concat([
                pd.DataFrame({"time_stamp": t_buy,  "side": 1}),
                pd.DataFrame({"time_stamp": t_sell, "side": 0}),
            ], ignore_index=True).sort_values("time_stamp").reset_index(drop=True)

            numerous_dfs.append(df_sim)
    else:
        taille = len(numerous_dfs)

    # Evaluation Loop
    for k in range(taille):
        current_df = numerous_dfs[k]

        # Optimization SLSQP
        resultat = minimize(
            fun=neg_log_likelihood_bivariate,
            x0=theta_init,
            args=(df_sim,),
            method='trust-constr',  # We try another algorithm : which is more robust to
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'xtol': 1e-8, 'gtol': 1e-8, 'disp': True}
        )

        # Two vectors of residuals
        u1, u2 = hawkes_residuals_bivariate(resultat.x, current_df)

        # Test on dim=1 (Buy)
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

        # Test on dim=2 (Sell)
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

        # Strict Global Test (everything needs to pass)
        if (pass_ks_1 and pass_lb_1 and pass_er_1) and (pass_ks_2 and pass_lb_2 and pass_er_2):
            count_global_joint += 1

    # Every Pass_Rates
    if verbose:
        print(f"\n--- PASS RATE RESULTS (out of {taille} simulations) ---")
        print("\n[Dimension 1 - Buys]")
        print(f"KS : {(count_ks_1 / taille) * 100:.2f}% | LB : {(count_lb_1 / taille) * 100:.2f}% | ED : {(count_er_1 / taille) * 100:.2f}%")
        print(f"Joint 1 : {(count_joint_1 / taille) * 100:.2f}%")

        print("\n[Dimension 2 - Sells]")
        print(f"KS : {(count_ks_2 / taille) * 100:.2f}% | LB : {(count_lb_2 / taille) * 100:.2f}% | ED : {(count_er_2 / taille) * 100:.2f}%")
        print(f"Joint 2 : {(count_joint_2 / taille) * 100:.2f}%")

        print("\n[Global Bivariate Model]")
        print(f"Strict Validation (Intersection of all 6 tests) : {(count_global_joint / taille) * 100:.2f}%")

    return {
        "ks_1": count_ks_1, "lb_1": count_lb_1, "er_1": count_er_1, "joint_1": count_joint_1,
        "ks_2": count_ks_2, "lb_2": count_lb_2, "er_2": count_er_2, "joint_2": count_joint_2,
        "global_joint": count_global_joint, "taille": taille,
    }


if __name__ == "__main__":

    # ----- Discrete Hawkes demos (cells 2-3) -----
    DHP("Exponential", 0.2, 0.3, "Poisson", 100, 0.01)

    DHP("Exponential", 0.05, 0.5, "Bernouilli", 1000, 0.01)
    DHP("Exponential", 0.10, 0.5, "Bernouilli", 1000, 0.01)
    DHP("Exponential", 0.20, 0.5, "Bernouilli", 1000, 0.01)
    DHP("Exponential", 0.30, 0.5, "Bernouilli", 1000, 0.01)

    # ----- Continuous Hawkes demos (cells 6-7) -----
    mu = 1.0    # baseline intensity
    gamma = 0.8    # amplitude du noyau
    beta = 1.0    # decay
    T_max = 50.0   # horizon

    CHP_plot(mu, beta, gamma, T_max)

    CHP_plot(mu=0.7, beta=5, gamma=1, T_max=50)
    CHP_plot(mu=1.0, beta=1.0, gamma=0.9, T_max=50)
    CHP_plot(mu=0.5, beta=1.0, gamma=0.95, T_max=20)
    CHP_plot(mu=1.0, beta=5.0, gamma=4.0, T_max=30)
    CHP_plot(mu=1.0, beta=0.3, gamma=0.24, T_max=100)
    CHP_plot(mu=0.2, beta=1.0, gamma=0.85, T_max=100)

    # ----- Confidence interval demos (cells 11-12) -----
    CHP_IC_plot(mu, beta, gamma, T_max)
    CHP_IC_plot(mu=0.4, beta=5, gamma=1, T_max=50)
    CHP_IC_plot(mu=1.0, beta=1.0, gamma=0.9, T_max=50)
    CHP_IC_plot(mu=0.5, beta=1.0, gamma=0.95, T_max=50)
    CHP_IC_plot(mu=1.0, beta=5.0, gamma=4.0, T_max=50)
    CHP_IC_plot(mu=1.0, beta=0.3, gamma=0.24, T_max=50)
    CHP_IC_plot(mu=0.2, beta=1.0, gamma=0.85, T_max=50)

    # ----- Univariate fit (cells 13-15) -----
    time_stamp = CHP(mu=1.0, beta=0.3, gamma=0.24, T_max=10000)
    resultat = fit_univariate(time_stamp)

    # ----- Univariate goodness of fit (cells 17-18) -----
    theta_estime = [1.0055, 0.7997, 0.2981]
    univariate_goodness_of_fit(theta_estime, time_stamp)

    # ----- Univariate pass-rate loop (cell 19) -----
    pass_rate_univariate(n_sim=10)

    # ----- Bivariate simulation (cell 20) -----
    df_sim = simulate_bivariate()
    print(df_sim.head())

    # ----- Bivariate fit : trust-constr (cell 24) -----
    resultat = fit_bivariate(df_sim, method='trust-constr')

    # ----- Bivariate fit : SLSQP (cell 25) -----
    resultat = fit_bivariate(df_sim, method='SLSQP')

    # ----- Bivariate residuals + goodness of fit (cells 27-28) -----
    theta_estimate = [resultat.x[i] for i in range(8)]
    bivariate_goodness_of_fit(theta_estimate, df_sim)

    # ----- Bivariate pass-rate loop (cell 29) -----
    pass_rate_bivariate(n_sim=10)