# Kaggle Notebook

- [Complete Guide on Time Series Analysis in Python](https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python/notebook)
- 
- 





# 分解趋势、季节效应

1、[提取季节效应](https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python?scriptVersionId=41657782&cellId=41)

```
# Subtracting the Trend Component


# Time Series Decomposition
result_mul = seasonal_decompose(df['Number of Passengers'], model='multiplicative', period=30)


# Deseasonalize
deseasonalized = df['Number of Passengers'].values / result_mul.seasonal


# Plot
plt.plot(deseasonalized)
plt.title('Air Passengers Deseasonalized', fontsize=16)
plt.plot()
```



2、[提取趋势效应](https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python?scriptVersionId=41657782&cellId=38)

```
# Using scipy: Subtract the line of best fit
from scipy import signal
detrended = signal.detrend(df['Number of Passengers'].values)
plt.plot(detrended)
plt.title('Air Passengers detrended by subtracting the least squares fit', fontsize=16)
```

