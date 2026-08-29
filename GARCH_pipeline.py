import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
from numpy.polynomial import Polynomial
from scipy.linalg import toeplitz
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.optimize import minimize
from scipy.special import gammaln




def simulate_GJR_GARCH_student(n, alpha0, alpha1, beta0, v, y, loi):
    if alpha0 <= 0 or alpha1 < 0 or beta0 < 0:
        raise ValueError("Les paramètres doivent être strictement positifs pour garantir une variance > 0.")
    if alpha1 + beta0 + y/2 >= 1:
        raise ValueError("Le processus n'est pas stationnaire (alpha1 + beta0 >= 1). La variance va exploser.")

    rng = np.random.default_rng()
    if loi=='normal':
        z = rng.standard_normal(n)
    else:
        z = rng.standard_t(df=v, size=n)*np.sqrt((v-2)/v)
    
    eps = np.zeros(n)
    sigma_carre = np.zeros(n)
    
    eps[0] = 0 #init
    sigma_carre[0] = alpha0 / (1 - alpha1 - beta0) # Variance inconditionnelle du GARCH
    
    for k in range(1, n):
        I_k = 1 if eps[k-1] < 0 else 0

        sigma_carre[k] = y*(eps[k-1]**2)*I_k + alpha0 + alpha1 * (eps[k-1]**2) + beta0 * sigma_carre[k-1]
        
        eps[k] = z[k] * np.sqrt(sigma_carre[k])
        
    return eps, sigma_carre

    #sigma_carre est la vol conditionelle

def AR_GJR_GARCH_student(n,phi0,phi1,alpha0,alpha1,beta0,v,y,loi):
    
    if abs(phi1) >= 1:
        raise ValueError("Paramètre AR invalide : le processus n'est pas stationnaire en moyenne.")
    
    if alpha0 <= 0 or alpha1 < 0 or beta0 < 0:
        raise ValueError("Les paramètres doivent être strictement positifs pour garantir une variance > 0.")
    if alpha1 + beta0 + y/2 >= 1:
        raise ValueError("Le processus n'est pas stationnaire (alpha1 + beta0 >= 1). La variance va exploser.")

    epsilon, vol_conditionnelle = simulate_GJR_GARCH_student(n, alpha0, alpha1, beta0,v,y,loi)

    actif = np.zeros(n)

    actif[0] = (phi0 / (1 - phi1)) + epsilon[0] #premier instant + choc

    for k in range(1, n):
        actif[k] = phi0 + phi1 * actif[k-1] + epsilon[k]

    print(f'AR(1)-GJR_GARCH(1) with Zt suivant une loi de Student sur {n} périodes avec comme paramètres θ= (phi0 = {phi0}, phi1 = {phi1}, alpha0 = {alpha0}, alpha1 = {alpha1})')
    plt.figure(figsize=(8, 5))
    plt.plot(actif)
    plt.title('AR(1)-GJR_GARCH(1)-Student-Law')
    plt.ylim(-0.06,0.06)
    plt.xlabel('Temps')
    plt.ylabel('Log-Rendement-Actif')
    plt.show()

    return epsilon,actif


def log_vraisemblance(zeta, actif,loi):

    if loi=='normal':
        return log_vraisemblance_GARCH_student(zeta, actif):
    else:
        return log_vraisemblance_GJR_GARCH_student(zeta, actif):

    phi0, phi1, alpha0, alpha1, beta0, v , y = zeta

    #on doit avoir au moins 2 degrés de libertés, doù le fait qu'on ajoute cette condition
    if alpha0 <= 0 or alpha1 < 0 or beta0 < 0 or (alpha1 + beta0+ y/2) >= 1 or abs(phi1) >= 1 or v <= 2.001:
        return 1e10

    T = len(actif)
    
    eps = np.zeros(T)
    eps[0] = actif[0] - (phi0 / (1 - phi1)) # Choc inconditionnel initial

    for t in range(1, T):
        eps[t] = actif[t] - phi0 - phi1 * actif[t-1] # permet de calculer tous les epsilons 

    sigma2_t_minus_1 = np.var(actif) # init variance conditionelle
    
    log_v = 0
    
    #on utilise gammaln sinon trop coûteux à calculer
    cst_student = gammaln((v + 1) / 2) - gammaln(v / 2) - 0.5 * np.log(np.pi * (v - 2)) #pre-calcul de la constante gamma, éviter trop de calcul
    
    for t in range(1, T):
        I_t = 1 if eps[t-1] < 0 else 0
        sigma2_t = y*(eps[t-1]**2)*I_t + alpha0 + alpha1 * (eps[t-1]**2) + beta0 * sigma2_t_minus_1 # maj de la variance avec eps
        
        log_v += cst_student - 0.5 * np.log(sigma2_t) - 0.5 * (v + 1) * np.log(1 + (eps[t]**2) / ((v - 2) * sigma2_t)) #on input
        
        sigma2_t_minus_1 = sigma2_t
    
    return -log_v


def opti_AR_GJR_GARCH_student(log_vraisemblance_GARCH_student,parametres_initiaux,actif):

    resultat_optimisation = minimize(
        fun=log_vraisemblance_GJR_GARCH_student, 
        x0=parametres_initiaux, 
        args=(actif), 
        method='Nelder-Mead' # Algorithme robuste qui ne nécessite pas de gradient parfait
    )

    phi0_opt, phi1_opt, alpha0_opt, alpha1_opt, beta0_opt , v_opt, y_opt = resultat_optimisation.x
    print(f"Paramètres optimaux : {resultat_optimisation.x}")
    print(f"Succès de la convergence : {resultat_optimisation.success}")
    return phi0_opt, phi1_opt, alpha0_opt, alpha1_opt, beta0_opt, v_opt, y_opt


