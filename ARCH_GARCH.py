import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
from numpy.polynomial import Polynomial
from scipy.linalg import toeplitz
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.optimize import minimize

def simulate_ARCH(n,alpha0,alpha1):
    x=0
    if alpha0 <= 0 or alpha1 < 0 or alpha1 >= 1:
        raise ValueError("Paramètres ARCH invalides : risque de variance négative ou explosive.")
    z=rng.standard_normal(n)
    sigma_carre= alpha0 / (1 - alpha1)

    resultats=[0]*(n+1)

    vol=[0]*(n+1)
    vol[0]=sigma_carre

    for k in range(0,n):
        x=z[k]*np.sqrt(sigma_carre)
        sigma_carre=alpha0+alpha1*(x**2)
        resultats[k+1]=x
        vol[k+1]=sigma_carre
    return resultats,vol

def estimation(t, n, alpha0, alpha1):
    total_vol = [0] * n
    total_kurto = [0] * n

    for k in range(n):
        x, vol = simulate_ARCH(t, alpha0, alpha1)

        x = np.array(x)

        # Moyenne empirique
        x_empirique = np.mean(x)

        # Variance empirique
        termes = (x - x_empirique) ** 2
        estimation = np.mean(termes)

        total_vol[k] = estimation

        # Kurtosis
        x_empirique_4 = np.mean((x - x_empirique)**4) #Au cas où ce n'est pas centré
        v_empirique_2 = estimation ** 2

        kurtosis_inc = x_empirique_4 / v_empirique_2

        total_kurto[k] = kurtosis_inc

    return total_vol, total_kurto


def AR(p,sigma,n):
    racines=1/rng.uniform(-1,1,p) #We first begin to create random roots

    P_unitaire = Polynomial.fromroots(racines) #We create our polynome using our roots (unitary polynome)
    P_final=P_unitaire

    coefficients = P_final.coef #Ce ne sont pas les phi attention (mais - phi)
    ar = P_final.coef / P_final.coef[0] #On force le polynome à être unitaire

    ma = np.array([1]) #On crée un MA constant (juste pour mettre l'argument dans ARMA Process)

    processus = ArmaProcess(ar,ma)

    gamma = processus.acovf(nobs=p) * (sigma) #Notre vecteur d'auto covariance
    matrice_covariance = toeplitz(gamma[:p]) # Matrice de covariance à l'aide de Toeplitz
    
    mean = np.zeros(p) 

    init=rng.multivariate_normal(mean,matrice_covariance)
    evo=[0]*n

    total = np.concatenate([init, evo])

    phis=-ar[1:] #On enlève la constante
    phis=phis[::-1] #On met dans l'ordre inverse


    for k in range(p,len(total)):
        bruit=rng.normal(0,sigma**2)
        x0=phis@total[k-p:k]+bruit
        total[k]=x0
    
    plt.figure(figsize=(8, 5))
    plt.plot(total)
    plt.title(f'AR({p}) sur {n+p} périodes et un écart-type {sigma}')
    plt.xlabel('Temps')
    plt.ylabel('Xt')
    plt.show()
    print('racines inverses : ',racines) 

def AR_ARCH(n,phi0,phi1,alpha0,alpha1):
    if abs(phi1) >= 1:
        raise ValueError("Paramètre AR invalide : le processus n'est pas stationnaire en moyenne.")
    
    epsilon,_= simulate_ARCH(n,alpha0,alpha1)
    mu=[0]*(n+1)

    mu[0]=phi0/(1-phi1)

    for k in range(0,n):
        mu[k+1]=phi0+phi1*(mu[k]+epsilon[k])

    epsilon=np.array(epsilon)
    mu=np.array(mu)
    actif=epsilon+mu
    print(f'AR(1)-ARCH(1) sur {n} périodes avec comme paramètres θ= (phi0 = {phi0}, phi1 = {phi1}, alpha0 = {alpha0}, alpha1 = {alpha1})')
    plt.figure(figsize=(8, 5))
    plt.plot(actif)
    plt.title('AR(1)-ARCH(1)')
    plt.xlabel('Temps')
    plt.ylabel('Rt')
    plt.show()

    return epsilon,mu,actif

def log_vraisemblance(theta,actif):
    T=len(actif)
    [phi0,phi1,alpha0,alpha1]=theta

    if alpha0 <= 0 or alpha1 < 0 or alpha1 >= 1 or abs(phi1) >= 1:
        return 1e10

    log_v=0
    for k in range(0,T-2):
        sigma_2_param=alpha0+alpha1*(actif[k+1]-phi0-phi1*actif[k])**2
        eps_2_param=(actif[k+2]-phi0-phi1*actif[k+1])**2

        log_v+=(-1/2)*np.log(2*np.pi)-(1/2)*np.log(sigma_2_param)-eps_2_param/(2*sigma_2_param) #On pourrait enlever la constante pour l'opti
    
    return -log_v

def opti_AR_ARCH(log_vraisemblance,parametres_initiaux,actif):

    resultat_optimisation = minimize(
        fun=log_vraisemblance, 
        x0=parametres_initiaux, 
        args=(actif), 
        method='Nelder-Mead' # Algorithme robuste qui ne nécessite pas de gradient parfait
    )

    phi0_opt, phi1_opt, alpha0_opt, alpha1_opt = resultat_optimisation.x
    print(f"Paramètres optimaux : {resultat_optimisation.x}")
    print(f"Succès de la convergence : {resultat_optimisation.success}")
    return phi0_opt, phi1_opt, alpha0_opt, alpha1_opt

def simulate_GARCH(n, alpha0, alpha1, beta0):
    if alpha0 <= 0 or alpha1 < 0 or beta0 < 0:
        raise ValueError("Les paramètres doivent être strictement positifs pour garantir une variance > 0.")
    if alpha1 + beta0 >= 1:
        raise ValueError("Le processus n'est pas stationnaire (alpha1 + beta0 >= 1). La variance va exploser.")

    rng = np.random.default_rng()
    z = rng.standard_normal(n)
    
    eps = np.zeros(n)
    sigma_carre = np.zeros(n)
    
    eps[0] = 0 #init
    sigma_carre[0] = alpha0 / (1 - alpha1 - beta0) # Variance inconditionnelle du GARCH
    
    for k in range(1, n):
        sigma_carre[k] = alpha0 + alpha1 * (eps[k-1]**2) + beta0 * sigma_carre[k-1]
        
        eps[k] = z[k] * np.sqrt(sigma_carre[k])
        
    return eps, sigma_carre

    #sigma_carre est la vol conditionelle

def AR_GARCH(n,phi0,phi1,alpha0,alpha1,beta0):
    
    if abs(phi1) >= 1:
        raise ValueError("Paramètre AR invalide : le processus n'est pas stationnaire en moyenne.")
    
    if alpha0 <= 0 or alpha1 < 0 or beta0 < 0:
        raise ValueError("Les paramètres doivent être strictement positifs pour garantir une variance > 0.")
    if alpha1 + beta0 >= 1:
        raise ValueError("Le processus n'est pas stationnaire (alpha1 + beta0 >= 1). La variance va exploser.")

    epsilon, vol_conditionnelle = simulate_GARCH(n, alpha0, alpha1, beta0)

    actif = np.zeros(n)

    actif[0] = (phi0 / (1 - phi1)) + epsilon[0] #premier instant + choc

    for k in range(1, n):
        actif[k] = phi0 + phi1 * actif[k-1] + epsilon[k]

    print(f'AR(1)-GARCH(1) sur {n} périodes avec comme paramètres θ= (phi0 = {phi0}, phi1 = {phi1}, alpha0 = {alpha0}, alpha1 = {alpha1})')
    plt.figure(figsize=(8, 5))
    plt.plot(actif)
    plt.title('AR(1)-GARCH(1)')
    plt.xlabel('Temps')
    plt.ylabel('Rt')
    plt.show()

    return epsilon,actif


def log_vraisemblance_GARCH(zeta, actif):
    phi0, phi1, alpha0, alpha1, beta0 = zeta

    if alpha0 <= 0 or alpha1 < 0 or beta0 < 0 or (alpha1 + beta0) >= 1 or abs(phi1) >= 1:
        return 1e10

    T = len(actif)
    
    eps = np.zeros(T)
    eps[0] = actif[0] - (phi0 / (1 - phi1)) # Choc inconditionnel initial

    for t in range(1, T):
        eps[t] = actif[t] - phi0 - phi1 * actif[t-1] #permet de calculer tous les epsilons 

    sigma2_t_minus_1 = np.var(actif) #init variance conditionelle : on considère ici la variance empirique de nos données
    
    log_v = 0
    
    for t in range(1, T):
        sigma2_t = alpha0 + alpha1 * (eps[t-1]**2) + beta0 * sigma2_t_minus_1 #maj de la variance avec eps
        
        log_v += -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2_t) - (eps[t]**2) / (2 * sigma2_t) #On input
        
        sigma2_t_minus_1 = sigma2_t
    
    return -log_v


def opti_AR_GARCH(log_vraisemblance_GARCH,parametres_initiaux,actif):

    resultat_optimisation = minimize(
        fun=log_vraisemblance_GARCH, 
        x0=parametres_initiaux, 
        args=(actif,), 
        method='Nelder-Mead' # Algorithme robuste qui ne nécessite pas de gradient parfait
    )

    phi0_opt, phi1_opt, alpha0_opt, alpha1_opt, beta0_opt  = resultat_optimisation.x
    print(f"Paramètres optimaux : {resultat_optimisation.x}")
    print(f"Succès de la convergence : {resultat_optimisation.success}")
    return phi0_opt, phi1_opt, alpha0_opt, alpha1_opt, beta0_opt