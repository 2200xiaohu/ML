# Kaggle Notebook

- [Complete Guide on Time Series Analysis in Python](https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python/notebook)
- [ARIMA](https://www.kaggle.com/code/prashant111/arima-model-for-time-series-forecasting)



# Blog

- [Darts with multiple](https://unit8.com/resources/training-forecasting-models/)
- [单序列](https://towardsdatascience.com/darts-swiss-knife-for-time-series-forecasting-in-python-f37bb74c126)







# 几种任务

## 多元时间序列

1、有一个target，其他是辅助的变量。并且辅助的变量可以预先测量的

可用模型例如：XGBoost、LGB、Theta、N-BEATS

2、所有序列都是target，需要同时预测所有序列

例如：VARMAX、N-BEATS



# 单序列时间分析

例如：ARIMA、SARIMAX、Theta、facebook Prophet、N-BEATS



PS：[N-BEATS](https://towardsdatascience.com/n-beats-time-series-forecasting-with-neural-basis-expansion-af09ea39f538)

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







