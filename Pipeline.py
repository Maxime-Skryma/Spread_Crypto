import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


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
    print(columns)

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
    plt.show()
    fig.savefig("Spread_Crypto/figs/crypto_df.png")
        
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
    plt.show()
    fig.savefig("Spread_Crypto/figs/crypto_log_df.png")

    #===================================================================
    # Manual ADF and Automatic ADF on (log) - currencies

    for ls in log_symbols:
        aics = {}
        log = df[ls]
        print(f"\n=== {ls} ===")
        for p in range(1, 16):
            t, aic, n = adf_manuel(log, p) #boucle pour appliquer l'ADF manuel pour différents nombre de lags
            aics[p] = aic #On remplit notre tableau d'AIC (pour pouvoir en tirer l'argmin notamment)
            ref = adfuller(log.dropna(), maxlag=p, autolag=None, regression='c')[0] #On applique directement notre ADF automatique sur les colonnes log du df[]
            print(f"p={p:2d}  t={t:7.3f}  refdf={ref:7.3f}  écart={abs(t-ref):.2e} AIC={aic:.2e} n={n}")
        p_opt = min(aics, key=aics.get)
        print(f'the number of diffs for {ls} that minimizes AIC is {p_opt}')
    with open("Spread_Crypto/results/ADF_summary.txt", "w") as f:
    f.write(str(ref))

    #===================================================================
    # Manual ADF and Automatic ADF on Diff - (log) - currencies
    for ls in log_symbols:
        aics = {}
        log = df[ls]
        print(f"\n=== {ls} ===")
        for p in range(1, 16):
            t, aic, n = adf_manuel(log.diff(), p) #boucle pour appliquer l'ADF manuel pour différents nombre de lags
            aics[p] = aic #On remplit notre tableau d'AIC (pour pouvoir en tirer l'argmin notamment)
            ref = adfuller(log.dropna(), maxlag=p, autolag=None, regression='c')[0] #On applique directement notre ADF automatique sur les colonnes log du df[]
            print(f"p={p:2d}  t={t:7.3f}  refdf={ref:7.3f}  écart={abs(t-ref):.2e} AIC={aic:.2e} n={n}")
        p_opt = min(aics, key=aics.get)
        print(f'the number of diffs for diff-{ls} that minimizes AIC is {p_opt}')
    with open("Spread_Crypto/results/Diff_ADF_summary.txt", "w") as f:
    

    #===================================================================
    # Automatic ADF on Res of regressions
    with open("Spread_Crypto/results/Res_ADF.txt", "w") as f:
    for ls1 in log_symbols:
        for ls2 in log_symbols:
            if ls1!=ls2:
                resultats=sm.OLS(df[ls1],sm.add_constant(df[ls2])).fit() #Modèle OLS avec constante
                df[f'Res of {ls1} on {ls2}']=resultats.resid #residuals of the OLS
                nom_colonne = f'Res of {ls1} on {ls2}' #On stocke le nom de la colonne
                t_stat = adfuller(df[nom_colonne], maxlag=p, autolag='AIC', regression='n')[0] #On applique un ADF sur le résidu
                f.write(f'{nom_colonne} has a t_value of {t_stat}\n')