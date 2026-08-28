import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.api import VAR
import numpy as np
import os





def adf_manuel(log, p): #log est notre série, p est le nombre de lags
    dlog = log.diff() #Δlog(X) il va notamment servir pour la variable à prédire, et les lags différenciés, mais c'est pas la composante principale
    X = pd.DataFrame(index=log.index) #Notre tableau des valeurs de log(X)
    X['niveau_lag1'] = log.shift(1) #On ajoute une colonne avec un lag 1 : ie Xt-1, ça sera ce sur quoi on régresse.
    for i in range(1, p + 1):
        X[f'dlog_lag_{i}'] = dlog.shift(i) #Ici on crée les lags différenciés à 'p niveau', Xt-2 - Xt-3 par exemple, Xt-4 - Xt-5
    X['const'] = 1.0 #constante pour notre régression
    reg = pd.concat([dlog.rename('dlog'), X], axis=1).dropna() #concaténation de X et de notre lag différencié (ce qu'on veut prédire)
    res = sm.OLS(reg['dlog'], reg.drop(columns='dlog')).fit() #on fit notre régression : on prédit Δlog(Xt) à l'aide de log(X t-1), Δlog(Xt-1), Δlog(Xt-2)...
    return res.tvalues['niveau_lag1'], res.aic, len(reg) #On demande d'avoir la t_value, uniquement du lag 1, c'est celle qui nous intéresse


def pipeline(df):
    #===================================
    # 1st Figure : Evolution of Currencies
    locator = mdates.AutoDateLocator()

    fig,ax=plt.subplots(figsize=(12,5))

    columns = pd.Series(df.columns) #On en fait une Série
    columns = df.columns[1:] #On enlève open_time

    for col in columns:
        ax.plot(df['open_time'],
        df[col]/(df[col].iloc[0])*100, #To be on the same scale, we compare their evolution, not their level
        label=col,
        alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized df in base 100")
    ax.legend()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.savefig("figs/crypto_df.png")
        
    #===================================
    log_symbols=[]
    for col in columns:
        log_symbols.append(f'log_{col}')
        df[f'log_{col}']=np.log(df[col])
        
    # 2nd Figure : Evolution of Log_Currencies
    fig,ax=plt.subplots(figsize=(12,5))

    for col in log_symbols:
        ax.plot(df['open_time'],
        df[col], #To be on the same scale, we compare their evolution, not their level
        label=col,
        alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Log df")
    ax.legend()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.savefig("figs/crypto_log_df.png")

    #===================================================================
    # Manual ADF and Automatic ADF on (log) - currencies
    with open("results/ADF_summary.txt", "w") as f:
        for ls in log_symbols:
            aics = {}
            log = df[ls]
            for p in range(1, 16):
                t, aic, n = adf_manuel(log, p) #boucle pour appliquer l'ADF manuel pour différents nombre de lags
                aics[p] = aic #On remplit notre tableau d'AIC (pour pouvoir en tirer l'argmin notamment)
                ref = adfuller(log.dropna(), maxlag=p, autolag=None, regression='c')[0] #On applique directement notre ADF automatique sur les colonnes log du df[]
            p_opt = min(aics, key=aics.get)
        f.write(f'the number of diffs for {ls} that minimizes AIC is {p_opt}')
        f.write(str(ref))

    #===================================================================
    # Manual ADF and Automatic ADF on Diff - (log) - currencies
    with open("results/Diff_ADF_summary.txt", "w") as f:
        for ls in log_symbols:
            aics = {}
            log = df[ls]
            f.write(f"\n=== {ls} ===\n")
            for p in range(1, 16):
                t, aic, n = adf_manuel(log.diff(), p) #boucle pour appliquer l'ADF manuel pour différents nombre de lags
                aics[p] = aic #On remplit notre tableau d'AIC (pour pouvoir en tirer l'argmin notamment)
                ref = adfuller(log.dropna(), maxlag=p, autolag=None, regression='c')[0] #On applique directement notre ADF automatique sur les colonnes log du df[]
                f.write(f"p={p:2d}  t={t:7.3f}  refdf={ref:7.3f}  écart={abs(t-ref):.2e} AIC={aic:.2e} n={n}\n")
            p_opt = min(aics, key=aics.get)
            f.write(f'the number of lags for diff-{ls} that minimizes AIC is {p_opt}\n')
    

    #===================================================================
    # Automatic ADF on Res of regressions
    with open("results/Res_ADF.txt", "w") as f:
        for ls1 in log_symbols:
            for ls2 in log_symbols:
                if ls1!=ls2:
                    resultats=sm.OLS(df[ls1],sm.add_constant(df[ls2])).fit() #Modèle OLS avec constante
                    df[f'Res of {ls1} on {ls2}']=resultats.resid #residuals of the OLS
                    nom_colonne = f'Res of {ls1} on {ls2}' #On stocke le nom de la colonne
                    t_stat = adfuller(df[nom_colonne], maxlag=p, autolag='AIC', regression='n')[0] #On applique un ADF sur le résidu
                    f.write(f'{nom_colonne} has a t_value of {t_stat}\n')
    
    with open("results/p_value_Res_ADF.txt", "w") as f:
        for ls1 in log_symbols:
            for ls2 in log_symbols:
                if ls1 != ls2:
                    _, pvalue, _= coint(df[ls1], df[ls2]) #we just want to keep p_values
                    f.write(f'{nom_colonne} has a t_value of {pvalue}\n')
    
    #===================================================
    #VAR representation / Eigen Values / Modules
    matrice_prix=df[log_symbols].to_numpy() 
    
    # Toutes les différences
    delta_Y = matrice_prix[1:] - matrice_prix[:-1] 

    max_lags = 20
    dim=len(log_symbols)
    # Boucle OLS
    with open("results/VAR(p).txt", "w") as f:
        for p in range(1, max_lags + 1):
            
            f.write(f"\n{'='*40}")
            f.write(f"ESTIMATION DU MODÈLE AVEC p = {p}\n")
            f.write(f"{'='*40}")
            
            # On sacrifie 'p' jours
            Y_t_1_niveaux = matrice_prix[p : -1]
            Y_cible = delta_Y[p:]
            
            # On crée la liste des retards pour ce 'p'
            liste_des_retards = [delta_Y[p-k : -k] for k in range(1, p + 1)]
            
            X_matrice = np.hstack([Y_t_1_niveaux] + liste_des_retards)
            X_avec_constante = sm.add_constant(X_matrice)
            
            # OLS
            coefficients_globaux, residus, rang, val_sing = np.linalg.lstsq(X_avec_constante, Y_cible, rcond=None)
            
            # Calcul des valeurs propres issues de la VECM
            matrice_Pi_brute = coefficients_globaux[1:dim+1, :]
            A = matrice_Pi_brute + np.eye(dim)
            
            eigen_values = np.linalg.eigvals(A)
            modules = np.abs(eigen_values)
            f.write(f"Modules des valeurs propres : {np.round(modules, 4)}\n")

    #===================================================
    # VAR representation : p_values
    with open("results/VAR(p)_p_values.txt", "w") as f:

        delta_columns=[f'Δ_{log_col}' for log_col in log_symbols]
        df_rendements = pd.DataFrame(delta_Y, columns=delta_columns)

        modele = VAR(df_rendements)

        p = 15
                    
        for k in range(3,p+1):
            resultats = modele.fit(k)
            matrice_pvalues = resultats.pvalues

            f.write("="*40)
            f.write(f"La matrice des p_values pour {k} lags est\n")
            f.write("="*40)
            f.write(str(matrice_pvalues))
            
            matrice_correlation = resultats.resid.corr()
            f.write("="*40)
            f.write(f"La matrice de corrélation pour {k} lags est\n")
            f.write("="*40)
            f.write(str(matrice_correlation))

    #===================================================
    # Selection_retards : p_values
    with open("results/Selection_retards.txt", "w") as f:
        
        selection_retards = modele.select_order(maxlags=20)
        f.write(str(selection_retards.summary()))

    #===================================================
    # Granger test on optimal p according to AIC
  
    p = selection_retards.aic
    resultats = modele.fit(p) 
    #On choisit le p qui minimise AIC


    with open("results/Granger_Test.txt", "w") as f:
        for ls1 in df_rendements.columns:
            for ls2 in df_rendements.columns:
                if ls1 != ls2:
                    test_granger = resultats.test_causality(ls1,ls2, kind='f')
                    f.write(f"Granger Test of {ls1} on {ls2}\n")
                    f.write("="*40)
                    f.write(str(test_granger.summary()))


    #===================================================
    # IRF
    irf = resultats.irf(24) #analyse IRF sur 24h

    fig = irf.plot(orth=True) 
    fig.suptitle("Fonctions de Réponse Impulsionnelle (Chocs de Cholesky)", fontsize=16)
    fig.tight_layout()
    fig.savefig("figs/IRF.png")

    #===================================================
    # IRF but with Bootstrap (Monte-Carlo)
    fig = irf.plot(
        orth=True, 
        stderr_type='mc',  # 'mc' pour Monte-Carlo
        repl=1000,         # Nombre de simulations : 1000
        seed=42
    )

    fig.suptitle("IRF avec Intervalles de Confiance Bootstrap (1000 simulations)", fontsize=16)
    fig.tight_layout()
    fig.savefig("figs/IRF_BootStrapped.png")


    

        