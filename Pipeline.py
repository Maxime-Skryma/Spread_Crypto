import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#===================================
# 1st Figure : Evolution of Currencies
locator = mdates.AutoDateLocator()


fig,ax=plt.subplots(figsize=(12,5))

columns = pd.Series(prices.columns) #On en fait une Série
columns = prices.columns[1:] #On enlève open_time
print(columns)

for col in columns:
    ax.plot(prices['open_time'],
    prices[col]/(prices[col].iloc[0])*100, #To be on the same scale, we compare their evolution, not their level
    label=col,
    alpha=0.8)
ax.set_xlabel("Time")
ax.set_ylabel("Normalized Prices in base 100")
ax.legend()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.show()
    
#===================================
# 1st Figure : Evolution of Currencies

