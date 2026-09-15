import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


# Price dynamics: arithmetic Brownian motion (Avellaneda-Stoikov),
# not geometric

def simulate_bm(T, n, n_paths, s, sigma):
    dt = T / n  # time step: [0, T] partitioned into n intervals

    dW = sigma * np.sqrt(dt) * np.random.randn(n_paths, n)  # independent increments ~ N(0, sigma^2 * dt)

    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)  # cumsum reconstructs the path: W_tk = sum of dW_j (telescoping sum)

    W = W + s  # initial asset price at s
    return W


def simulate_bm_plot(T, n, n_paths,s,sigma):
    W=simulate_bm(T, n, n_paths,s,sigma)

    t = np.linspace(0, T, n + 1) # time span
    for k in range(0,n_paths):
        plt.plot(t,W[k])
    plt.xlabel('t')
    plt.ylabel('Wt')
    plt.title('Brownian Motion')
    plt.show()


#Let's try to calculate the benchmark strategy ? 

def simule_PNL_benchmark(T,dt,sigma,q,gamma,k,A,s):
    
    n=int(T/dt)
    t=np.linspace(0,T,n+1)

    w=0 # init of wealth
    
    
    wealth=[0]*(n+1) # for wealth evolution
    p_a=[0]*(n+1) # for ask price evolution
    p_b=[0]*(n+1) #for bid price evolution

    spread=gamma*(sigma**2)*(T-t)+(2/gamma)*np.log(1+(gamma/k)) # δa + δb 

    mid_price=simulate_bm(T, n, 1,s,sigma)[0]


    for i in range(n+1):

        delta_a= (spread[i]/2) #strategy benchmark : symetric !
        delta_b= (spread[i]/2)

        lambda_a_delta_a=A*np.exp(-k*delta_a) # λ(δa) : our arrival rate is a function of δa (completely normal considering that price affect sales)
        lambda_b_delta_b=A*np.exp(-k*delta_b) # λ(δb) 

        a=np.random.binomial(1,1-np.exp(-lambda_a_delta_a*dt)) #we use 1-exp(-λ(δa)) rather than λ(δa) (which could be >1)
        b=np.random.binomial(1,1-np.exp(-lambda_b_delta_b*dt))

        q= q+b-a # increase by 1 with probability λ(δa)*dt and decrease by 1 with probability λ(δb)*dt

        w = w + (s+delta_a)*a - (s-delta_b)*b
        
        #=== To follow the evolutions
        wealth[i]=w

        p_a[i]=mid_price[i]+delta_a
        p_b[i]=mid_price[i]-delta_b
    

    
    return wealth,p_a,p_b,mid_price,q


    


def simule_PNL_benchmark_plot(T,dt,sigma,q,gamma,k,A,s):
    n=int(T/dt)
    t=np.linspace(0,T,n+1)
    wealth,p_a,p_b,mid_price,q=simule_PNL_benchmark(T,dt,sigma,q,gamma,k,A,s)

    plt.plot(t,mid_price,color='black',label='Mid-market price')
    plt.scatter(t,p_a,facecolors='red', s=15,label='Price asked')
    plt.scatter(t,p_b,facecolors='none',edgecolors='red',s=20, label='Price bid')
    plt.xlabel("t")
    plt.legend()
    plt.show()

    plt.plot(t,wealth,color='orange',label='wealth')
    plt.xlabel("t")
    plt.title("Wealth evolution")
    plt.legend()
    plt.show()


#Now let's simulate it's strategy 

def simule_PNL_AS(T,dt,sigma,q,gamma,k,A,s):
    
    n=int(T/dt)
    t=np.linspace(0,T,n+1)

    w=0 #initialisation de la wealth
    
    
    wealth=[0]*(n+1) #tableau de suivi de la wealth
    p_a=[0]*(n+1) #tableau de suivi des prix ask
    p_b=[0]*(n+1) #tableau de suivi des prix bid
    res_price_evo=[0]*(n+1) #tableau de suivi du prix de réservation

    spread=gamma*(sigma**2)*(T-t)+(2/gamma)*np.log(1+(gamma/k)) # δa + δb 

    mid_price=simulate_bm(T, n, 1,s,sigma)[0]


    for i in range(n+1):

        reserve_price=mid_price[i]-q*gamma*(sigma**2)*(T-t[i]) #valorisation de l'actif selon l'agent à la période k

        delta_a= (spread[i]/2) - q*gamma*(sigma**2)*(T-t[i]) # -q*gamma*(sigma**2)*(T-t) = reserve_price - mid_price
        delta_b= (spread[i]/2) + q*gamma*(sigma**2)*(T-t[i]) # +q*gamma*(sigma**2)*(T-t) = mid_price - reserve_price

        lambda_a_delta_a=A*np.exp(-k*delta_a) # λ(δa) : our arrival rate is a function of δa (completely normal considering that price affect sales)
        lambda_b_delta_b=A*np.exp(-k*delta_b) # λ(δb) 

        a=np.random.binomial(1,1-np.exp(-lambda_a_delta_a*dt)) #we use 1-exp(-λ(δa)) rather than λ(δa) (which could be >1)
        b=np.random.binomial(1,1-np.exp(-lambda_b_delta_b*dt))

        q= q+b-a # increase by 1 with probability λ(δa)*dt and decrease by 1 with probability λ(δb)*dt

        w = w + (s+delta_a)*a - (s-delta_b)*b
        
        #=== To follow the evolutions
        wealth[i]=w

        p_a[i]=mid_price[i]+delta_a
        p_b[i]=mid_price[i]-delta_b

        res_price_evo[i]=reserve_price
    

    
    return wealth,p_a,p_b,mid_price,res_price_evo,q


def simule_PNL_AS_plot(T,dt,sigma,q,gamma,k,A,s):
    n=int(T/dt)
    t=np.linspace(0,T,n+1)
    wealth,p_a,p_b,mid_price,res_price_evo,q = simule_PNL_AS(T,dt,sigma,q,gamma,k,A,s)

    plt.plot(t,mid_price,color='black',label='Mid-market price')
    plt.plot(t,res_price_evo,color='green',label='Indifference price')
    plt.scatter(t,p_a,facecolors='red', s=15,label='Price asked')
    plt.scatter(t,p_b,facecolors='none',edgecolors='red',s=20, label='Price bid')
    plt.xlabel("t")
    plt.legend()
    plt.show()

    plt.plot(t,wealth,color='orange',label='wealth')
    plt.xlabel("t")
    plt.title("Wealth evolution")
    plt.legend()
    plt.show()


def simulation_comparison(nb_simulations,T,dt,sigma,q,gamma,k,A,s):

    profit_bench_total=[0]*nb_simulations
    profit_AS_total=[0]*nb_simulations

    profit_bench=0 #init
    profit_AS=0

    q_bench_total=[0]*nb_simulations
    q_AS_total=[0]*nb_simulations


    for i in range(nb_simulations):
        wealth_bench,_,_,mid_price_bench,q_bench=simule_PNL_benchmark(T,dt,sigma,q,gamma,k,A,s)
        wealth_AS,_,_,mid_price_AS,_,q_AS=simule_PNL_AS(T,dt,sigma,q,gamma,k,A,s)

        profit_bench=wealth_bench[-1]+mid_price_bench[-1]*q_bench #profit = wealth + inventaire * mid_price
        profit_AS=wealth_AS[-1]+mid_price_AS[-1]*q_AS
        
        profit_bench_total[i]=profit_bench
        profit_AS_total[i]=profit_AS

        q_bench_total[i]=q_bench
        q_AS_total[i]=q_AS

    U_sym = np.mean(profit_bench_total) - (gamma/2)*np.var(profit_bench_total,ddof=1)
    U_inv = np.mean(profit_AS_total) - (gamma/2)*np.var(profit_AS_total,ddof=1)

    print("Inventory Strategy :")
    print("="*40)
    print(f"Profit has a mean of {np.mean(profit_AS_total):.3f} and a std of {np.std(profit_AS_total, ddof=1):.3f}")
    print(f"Final q has a mean of {np.mean(q_AS_total):.3f} and a std of {np.std(q_AS_total, ddof=1):.3f}")

    print()

    print("Symetric :")
    print("="*40)
    print(f"Profit has a mean of {np.mean(profit_bench_total):.3f} and a std of {np.std(profit_bench_total, ddof=1):.3f}")
    print(f"Final q has a mean of {np.mean(q_bench_total):.3f} and a std of {np.std(q_bench_total, ddof=1):.3f}")

    print()
    print(f'the ratio of profit (Symetric/Inventory) ={np.mean(profit_bench_total)/np.mean(profit_AS_total):.3f} et ratio of the std {np.std(profit_bench_total, ddof=1)/np.std(profit_AS_total, ddof=1):.3f}')
    
    print("Finally :")
    print("="*40)
    print(f"Utility symmetric : {U_sym:.3f}")
    print(f"Utility inventory : {U_inv:.3f}")
    
    return profit_bench_total,profit_AS_total


def simulation_comparison_plots(nb_simulations,T,dt,sigma,q,gamma,k,A,s):

    profit_bench_total,profit_AS_total=simulation_comparison(nb_simulations,T,dt,sigma,q,gamma,k,A,s)

    plt.hist(profit_bench_total, bins='auto', color='none',
         edgecolor='grey', label="Symetric")

    plt.hist(profit_AS_total, bins='auto', color='red',
         alpha=0.5, label="Inventory Strategy")

    plt.title("Profit Distribution")
    plt.legend()
    plt.show()


#Now let's simulate it's strategy 


def simule_PNL_AS_Hawkes(T,beta,omega,sigma,q,gamma,k,A,s):

    if omega/beta >= 1:
        raise ValueError('Attention il faut que omega/beta < 1')
        
    w=0 #initialisation de la wealth
    
    phi=0
    
    PnL=[0] #tableau de suivi de la wealth
    p_a=[] #tableau de suivi des prix ask
    p_b=[] #tableau de suivi des prix bid

    mid_price=[s]
    mid_courant=s

    event_a=[]
    event_b=[]
    arrive=0
    arrive_total=[0]
    
    res_price_evo=[mid_courant-q*gamma*(sigma**2)*(T-arrive)] #tableau de suivi du prix de réservation



    while True:
        
        def delta_a_eval(x):
            return (gamma*(sigma**2)*(T-x)+(2/gamma)*np.log(1+(gamma/k)))/2 - q*gamma*(sigma**2)*(T-x)

        def delta_b_eval(x):
            return (gamma*(sigma**2)*(T-x)+(2/gamma)*np.log(1+(gamma/k)))/2 + q*gamma*(sigma**2)*(T-x)

        u=np.random.uniform(0,1)
        v=np.random.uniform(0,1)
        ### On a recalculé analytiquement notre fonction de répartition mais il sera nécessaire de vérifier nos calculs

        F_a = lambda tau: 1- np.exp(-( A/(k*gamma*(sigma**2)*(1/2-q)) * (np.exp(-k*delta_a_eval(arrive+tau)) - np.exp(-k*delta_a_eval(arrive))) + (phi/beta)*(1-np.exp(-beta*(tau)))))
        F_b = lambda tau: 1- np.exp(-( A/(k*gamma*(sigma**2)*(1/2+q)) * (np.exp(-k*delta_b_eval(arrive+tau)) - np.exp(-k*delta_b_eval(arrive))) + (phi/beta)*(1-np.exp(-beta*(tau)))))


        #Trouver une borne analytiquement, 100 est arbitraire
        try:
            G_delta_a = brentq(lambda tau: F_a(tau) - u, 0, T - arrive)
        except ValueError:
            G_delta_a = np.inf
        
        try:
            G_delta_b = brentq(lambda tau: F_b(tau) - v, 0, T - arrive)
        except ValueError:
            G_delta_b = np.inf

        G_delta = min(G_delta_a,G_delta_b)
        arrive = arrive + G_delta

        if arrive>T:
            break


        mid_price.append(mid_courant + sigma * np.sqrt(G_delta) * np.random.randn())
        mid_courant=mid_price[-1]

        if G_delta_a<G_delta_b:
            event_a.append(arrive)
            q = q-1
            p_a.append(delta_a_eval(arrive) + mid_courant) #on lui ajoute le prix de vente
            w = w + delta_a_eval(arrive) + mid_courant
        else:
            event_b.append(arrive)
            q= q+1
            p_b.append(mid_courant - delta_b_eval(arrive))
            w = w - mid_courant + delta_b_eval(arrive)

        
        reserve_price=mid_courant-q*gamma*(sigma**2)*(T-arrive) #valorisation de l'actif selon l'agent à la période k

        phi=phi*np.exp(-beta*G_delta) + omega  # Φn+1 = Φn * exp(-β * Δn+1) + omega


        #=== To follow the evolutions

        arrive_total.append(arrive)
        PnL.append(w+ q*mid_courant)
        res_price_evo.append(reserve_price)
    

    
    return PnL,p_a,p_b,mid_price,res_price_evo,q,arrive_total,event_a,event_b


def simule_PNL_AS_Hawkes_plot(T, beta, omega, sigma, q, gamma, k, A, s):
    PnL, p_a, p_b, mid_price, res, q ,arrive_total,event_a,event_b= simule_PNL_AS_Hawkes(T, beta, omega, sigma, 0, gamma, k, A, s)


    plt.plot(arrive_total,mid_price,color='black',label='Mid-market price')
    plt.plot(arrive_total,res,color='green',label='Indifference price')
    plt.scatter(event_a,p_a,facecolors='red', s=15,label='Price asked')
    plt.scatter(event_b,p_b,facecolors='none',edgecolors='red',s=20, label='Price bid')
    plt.xlabel("t")
    plt.legend()
    plt.show()

    plt.plot(arrive_total,PnL,color='orange',label='wealth')
    plt.xlabel("t")
    plt.title("Wealth evolution")
    plt.legend()
    plt.show()


def simulation_Hawkes(nb_simulations,dt,T, beta, omega, sigma, q, gamma, k, A, s):

    profit_total=[0]*nb_simulations
    profit_total=[0]*nb_simulations

    q_total=[0]*nb_simulations


    for i in range(nb_simulations):
        PnL, p_a, p_b, mid_price, res, q ,arrive_total,event_a,event_b= simule_PNL_AS_Hawkes(T, beta, omega, sigma, 0, gamma, k, A, s)
        profit_total[i]=PnL[-1]

        q_total[i]=q


    U_inv = np.mean(profit_total) - (gamma/2)*np.var(profit_total,ddof=1)

    print("Inventory Strategy with Hawkes Process :")
    print("="*40)
    print(f"Profit has a mean of {np.mean(profit_total):.3f} and a std of {np.std(profit_total, ddof=1):.3f}")
    print(f"Final q has a mean of {np.mean(q_total):.3f} and a std of {np.std(q_total, ddof=1):.3f}")


    print("Finally :")
    print("="*40)
    print(f"Utility inventory : {U_inv:.3f}")
    
    return profit_total


def simulation_Hawkes_plots(nb_simulations,T,dt,beta,omega,sigma,q,gamma,k,A,s):

    profit_total=simulation_Hawkes(nb_simulations,dt,T, beta, omega, sigma, q, gamma, k, A, s),

    plt.hist(profit_total, bins='auto', color='red', label="Inventory")

    plt.title("Profit Distribution")
    plt.legend()
    plt.show()
