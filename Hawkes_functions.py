#Now let's simulate it's strategy 
#Let's first try to modelise a Discrete Hawkes process 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def DHP(kernel,gamma,theta,law,n,alpha0):
    if kernel=="Erlang":
        alpha=np.array([gamma*np.exp(-k*theta)*k for k in range(0,n)])
    if kernel=="Exponential":
        alpha=np.array([gamma*np.exp(-k*theta) for k in range(0,n)])
    
    alpha[0]=alpha0
    X_values=np.zeros(n)
    Sn=np.zeros(n)


    if law=="Bernouilli":
        X_values[0]=np.random.binomial(1,alpha[0])
        Sn[0]=X_values[0]
        for k in range(1,n):
            X_values[k]=np.random.binomial(1,alpha[0]+np.sum(alpha[k-1:0:-1]*X_values[1:k]))
            Sn[k]=Sn[k-1]+X_values[k]
    if law=="Poisson":
        X_values[0]=np.random.poisson(alpha[0])
        Sn[0]=X_values[0]
        for k in range(1,n):
            X_values[k]=np.random.poisson(alpha[0]+np.sum(alpha[k-1:0:-1]*X_values[1:k]))
            Sn[k]=Sn[k-1]+X_values[k]


    mu=alpha0/(1+alpha[0]-np.sum(alpha)) #on veut pas d'alpha0 or np.sum(alpha) l'inclut

    if law=="Bernouilli":
        sigma_asy=mu*(1-mu)/((1+alpha[0]-np.sum(alpha))**2) #same
    if law=="Poisson":
        sigma_asy=mu/((1+alpha[0]-np.sum(alpha))**2) #same

    Sn_div_n=[Sn[k]/(k+1) for k in range(0,n)]
    sigma_div=np.array([sigma_asy/np.sqrt(k+1) for k in range(0,n)])

    upper=mu + 1.96*sigma_div
    lower=mu - 1.96*sigma_div
    n_values=np.arange(1,n+1)


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

#Let's try to code a continuous Hawkes Process
# First let's try to just inverse the distribution function
#Then we will use the Thinning d'Ogatta


#We can define it so easily and recursively just because we have an exponential kernel, which allows to just multiply each time
#For calculation, I'll add a note on how we came to such an easy loop !


def CHP(alpha0,beta,gamma,T_max):

    if gamma/beta>=1:
        raise ValueError("Attention, il est nécessaire d'avoir : γ/β < 1 afin d'assurer la condition sur la fonction d'excitation")

    #alpha 0 : intensité de fond (baseline)
    #gamma : amplitude du noyau d'excitation
    #beta : taux de décroissance

    T=0
    phi=0

    N=[]

    phi_total=[0]
    while True:
        u=np.random.uniform(0,1)
        F = lambda delta: 1 -(np.exp(-alpha0*delta - phi*(1-np.exp(-beta*delta))/beta)) #Fonction de répartition de Δn+1 appliqué en δ (on part de la fonction de survie)

        b = -np.log(1 - u) / alpha0 + 1e-8   # borne haute garantie (car S(b) >= 1-e^{-alpha0*b} = u) : résultat analytique (see the note) 
        #On utilise cette borne haute pour être sûr que S(b)>0 et donc que l'algo brentq (qui nécessite au moins une annulation sur l'intervalle fonctionne)

        G_delta = brentq(lambda x: F(x) - u, 0, b) #G_delta est une simulation de la variable aléatoire Δn+1 

        T=T+ G_delta #Tn+1 = Tn + Δn+1   (ici G_delta car on utilise une simulation)
        if T > T_max:
            break #Permet de sortir de la boucle dès que le T calculé dépasse le T_max fixé

        phi=phi*np.exp(-beta*G_delta) + gamma #Φn+1 = Φn * exp(-β * Δn+1) + γ

        
        phi_total.append(phi)
        N.append(T) #N est totalement encodé par (T1,...,Tn)


    N = np.array(N)
    
    return N,phi_total

