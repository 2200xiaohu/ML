# 介绍

介绍一个不错的AutoML库，[autogluon](https://auto.gluon.ai/stable/index.html)。该库在Github上标星5.4k，并且有着详细的文档支持，上手迅速



# Install

以conda 虚拟环境为例，直接pip或conda 安装即可。安装比较慢，因为需要安装非常的依赖包

注意：仅支持pyhton>=3.8

```
pip install autogluon

conda install autogluon
```





# 快速使用

autogluon中主要有三个模块：1、Tabular Prediction；；3、Time Series Forecasting

1、Tabular Prediction模块：适合表格数据，即结构化数据

2、Multimodal Prediction：适合混合类型数据，支持文本和表格数据混合，也可以支持图像数据

3、Time Series Forecasting：适合时序数据



下面以Tabular Prediction模块为例，选取的是Kaggle上经典的Titanic比赛

[主要参考官方文档](https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-quickstart.html)



1、导入数据

非常随机的划分Train和Test

数据为结构化数据，但列中既有文本数据，也有数值类型。

```
df = pd.read_csv('titanic/train.csv')
train = df.sample(frac=0.7)
test = df.drop(df.sample(frac=0.7).index)
```



2、训练模型

主要代码： **TabularPredictor().fit()**

- [ ] 函数会自动检查数据中的分类变量、文本数据等等，自动进行处理
- [ ] 最后生成多个模型，同时会对多个做一个Stacking提供效果
- [ ] 训练好的模型会保存到“save_path”下，模型为pkl文件

注意：fit()函数其实有许多参数的，能够满足不同的需求，详情参考官方文档

```
label = 'Survived'#定义Lable，即目标变量
save_path = 'agModels-predictClass'#模型保存路径

#训练
predictor = TabularPredictor(label=label, path=save_path).fit(train,presets='best_quality')
#'best_quality'：选择精度最高的训练方法，但时间开销会变大
```



3、预测

```
#定义test
y_test = test[label] 
test_data_nolab = test.drop(columns=[label])  # delete label column to prove we're not cheating
test_data_nolab.head()

#从save_path导入训练好的模型
# unnecessary, just demonstrates how to load previously-trained predictor from file
predictor = TabularPredictor.load(save_path)  

#预测
y_pred = predictor.predict(test_data_nolab)
print("Predictions:  \n", y_pred)

#查看效果
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
```



4、综合查看效果

```
#使用训练好的模型，调用.leaderboard，查看在test上的效果
predictor.leaderboard(test, silent=True)
```

结果如下

![image-20230313161110386](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20230313161110386.png)

注意：stack_level表示在Stacking中的位置，可见函数做了给两层的Stacking，第二层模型为WeightedEnsemble_L2



5、最终得分

提交预测结果，查看在Kaggle上的得分

==最终得分：0.7799==

```
pre_data = pd.read_csv('titanic/test.csv')
predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(pre_data)

result = pd.DataFrame(pre_data['PassengerId'])
result['Survived'] = y_pred

#结果保存为submission.csv文件
result.to_csv('titanic/submission.csv',index=False)
```



# 总结

autogluon是一个不错的automl库，上手非常的简单，适合快速建模的需求。如果想要更仔细的使用，可以[参考官方文档](https://auto.gluon.ai/stable/index.html)，里面有更多的参数选择，例如选择何种方法训练模型等等