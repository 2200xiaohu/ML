# 管理你的ML model（By Neptune）使用例子：sklearn、LightGBM，Xgboost



## 目录：

**1、什么是Neptune**

**2、基于sklearn中的基本模型、LightGBM、XGboost展示**

**3、总结**





## 什么是Neptune？

**你是否还在为：模型可视化、模型performance可视化、学习曲线生成、不同模型比较等等，一系列繁杂的操作而烦恼？选择Neptune，这些问题都会迎刃而解**

昨天，我在研究如何使用LightGBM的时候，在Google的搜索结果中，发现了Neptune，给我打开了一个新世界的大门。

有过机器学习项目实战经验的同学们都知道，对于模型performance的检验和可视化是一个无比重要，但又繁杂的工作。尤其是这个模型是基于Cross Validation或者像LightGBM，Xgboost一样的集成模型时候。

可是如果你在你的代码中加入Neptune后，你会发现，这些问题它都帮你解决了，并且这一过程是十分方便。这就是Neptune的魅力！

<u>**Neptune的网址：https://neptune.ai/**</u>

![image-20220825195536148](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20220825195536148.png)

------



## 基于sklearn中的基本模型、LightGBM、XGboost展示

注意：

1、下列模型都不涉及超参数寻优。超参数寻优的话，sklearn的GridSearchCV不支持，但支持scikit-optimize

2、数据集基于Kaggle中的Titanic数据集

3、运行环境选择的是Google的“colab notebook"；你也可以在本地运行

4、x_train：训练集自变量；y_train：训练集因变量（目标变量）；x_test：测试集自变量；y_test：测试集因变量（目标变量）

------



### sklearn中的SVM

参考官方文档：https://docs.neptune.ai/integrations-and-supported-tools/model-training/sklearn

根据笔者在官方文档中查看到的资料，发现：现在暂且不支持sklearn中的超参数调参的function，例如GridSearchCV（）；但是，经过尝试发现，GridSearchCV（）也是可以用的，只是保存的performance等信息，与单一学习器（模型）相同，并没有任何优势。**证明：如果想运用Neputne监控超参数调参的话，sklearn中的函数还不支持，建议使用scikit-optimize**

#### 1、安装Neptune

```python
! pip install graphviz==0.10.1 neptune-client

pip install neptune-sklearn
```



#### 2、 激活 Neptune

1、运行代码后，下方会出现一个网址，点击网址即可查看你的model

2、在Neptune中的相关目录等级为：

**Account**

​	**|——>Project**

​					**|——>model**

3、查看自己的API参考官方文档：https://docs.neptune.ai/getting-started/installation#authentication-neptune-api-token

```python
import neptune.new as neptune
run = neptune.init(
      project="xiaohu2200/pratice",#我在Neptune注册的账号叫做xiaohu2200，并且我创建了一个project叫做pratice
      api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiM2RlZmNiNy0yNGMxLTRjNDAtOTBmZC04NWY3Y2MxMGZmZDYifQ==",#你可以在Neptune的个人账户界面查到自己的API
      name="svm_titanic",
      tags=["SVM", "notebook"],#自己添加的tag
  )
```



#### 3、运行SVM

**记住：一个模型的存储，需要先运行”run“，即上面的代码；一个run对应一个你要保存的model**

```python
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
start=time.time()
cross_valid_scores = {}
parameters = {
    "C": 0.1,
    "kernel": "linear",
    "gamma": "scale",
}

model_svc = SVC(
    random_state=2200,
    class_weight="balanced",
    probability=True,
)


model_svc.fit(x_train, y_train)
```

现在你运行的SVM模型就已经保存在你的project下了！

#### 4、创建一个summary记录模型的performance

这一步非常重要！没有这一步，无法得到模型的performance

```python
import neptune.new.integrations.sklearn as npt_utils

run["cls_summary"] = npt_utils.create_classifier_summary(model_svc, x_train, x_test, y_train, y_test)
```



#### 5、停止run

```python
run.stop()
```



#### 6、查看一下效果

点开网页，可以看到如下图：



![image-20220825203938503](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20220825203938503.png)



放大一点，仔细查看，可以看到，我们刚刚创建的**cls_summary文件夹**



![image-20220825203953215](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20220825203953215.png)



再点开他，就可以查看到我们模型的performance啦！



![image-20220825204121752](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20220825204121752.png)



当然，你可以直接在最初页面的**”Images“**，查看得到的所有图表

#### 总结

1、支持的不全面，只能查看基本的performance，缺少learning curve；

2、不支持sklearn里面的GridSearchCV（）进行超参数调参，但支持scikit-optimize进行超参数调参



------



### LightGBM

众所周知，像是LightGBM、Xgboost这些集成算法，有个最大问题是，它们实际上类似于一个黑箱子。这使得对于最终模型的解释非常困难。但Neptune帮你解决了这个问题！

对于LIghtGBM的支持可就比sklearn全面的多啦！你不仅可以查看模型的效果、学习曲线、特征重要性等等，甚至还有LightGBM中所生成的所有weak learner(弱学习器)的模型可视化。通过Neptune，你能更好的解释你所生成的LightGBM模型。

因为大致的过程和sklearn相似，所以相似的部分就简略过掉，不同的地方会重点标出

网址：https://docs.neptune.ai/integrations-and-supported-tools/model-training/lightgbm



#### 1、安装、激活Neptune

与sklearn中的不同，我们需要创建一个”callback“，这个callback会传递给lgb.train()，这使得在运行这个function的时候，将运行过程传到Neptune中。

```python
! pip install graphviz==0.16 neptune-client neptune-lightgbm lightgbm==3.3.2
import lightgbm as lgb
import neptune.new as neptune
from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary
run = neptune.init(
    project="xiaohu2200/pratice",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiM2RlZmNiNy0yNGMxLTRjNDAtOTBmZC04NWY3Y2MxMGZmZDYifQ==",
    name="LightGBM",
    tags=["LightGBM","notebook"],
)
neptune_callback = NeptuneCallback(run=run)
```

#### 2、训练LightGBM

```python
#转换一下data type
cols = x_train.columns
x_train[cols]=x_train[cols].astype('float')
x_test[cols]=x_test[cols].astype('float')
y_train=y_train.astype('bool')
y_test=y_test.astype('bool')
# 转换为lightgbm必须的data type
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# Define parameters
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "num_class": 1,
    "metric": ["binary_logloss", "binary_error"],#设置你想要保存的指标
    "num_leaves": 21,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 12,
}
#训练模型
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_eval],
    early_stopping_rounds=50,
    valid_names=["training", "validation"],
    callbacks=[neptune_callback]
)
```

#### 3、保存summary

```python
y_pred=gbm.predict(x_test)
y_pred[y_pred>0.5]=1
y_pred[y_pred<=0.5]=0
# Log summary metadata to the same run under the "lgbm_summary" namespace
run["lgbm_summary"] = create_booster_summary(
    booster=gbm,
    log_trees=True,#是否保存树
    list_trees=list(range(0,5)),#你想要保存多少颗树作为结果
    log_confusion_matrix=True,
    y_pred=y_pred,
    y_true=y_test,
)
run.stop()#停止run
```

#### 4、 查看结果

​	最重要的——lgbm_summary，里面存储了有：1、每个弱学习器的详细信息；2、混淆矩阵、特征重要性和弱学习器的模型可视化

​	（如下面的树形图）

![image-20220827145901755](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20220827145901755.png)

![image-20220827150212462](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20220827150212462.png)

除了这些，点开“charts”，你还可以查看整个模型的生成过程。横坐标表示弱学习器的个数，纵坐标表示error或者logloss（可以自己设置的）。通过这个曲线，你可以得到该超参数下，何时停止模型迭代效果最好。

![image-20220827150327002](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20220827150327002.png)



#### 5、其他LightGBM function

对于其他的LIghtGBM的function，可以看笔者写的colab notebook



#### 总结

1、Neptune支持对LightGBM 的performance（rmse，confusion matrix，Feature importance等等）查看，同时可以查看类似于Learning curve的图，以及模型可视化

2、对LightGBM的function：lgb.train(),lgb.cv()，lgb.LGBMClassifier支持，lgb.cv()不能生成summary,所以没有模型可视化

3、有效帮助进行模型可视化、超参数选择





### XGboost

因为XGboost的代码与LightGBM非常相似，所以就不再详细叙述了，只做总结

网址：https://docs.neptune.ai/integrations-and-supported-tools/model-training/xgboost

#### 总结

1、XGboost中的xgb.cv()提供的分析，比lgb.cv()更详细。包括有：每次fold的feature importance、模型可视化；不同弱学习器个数下所有fold的平均指标

2、XGboost和LightGBM的使用情况类似

3、只要是在sklearn的接口的话，建议将eval_metric参数直接在fit()中定义，不要设置在param中

## 写在最后

Neptune无疑是一个出色的公里模型的工具，并且它也十分容易上手。笔者的代码都是基于Google colab notebook而写的，里面的XGboost和LightGBM都是以及预装好的了。所以如果你在上手的时候，遇到一些问题，无法成功使用Neptune，可以试着将这两个包安装到最新的版本（例如LightGBM必须是3.2.2才能成功运行）。

我会将代码贴在这里，供大家查看：https://colab.research.google.com/drive/1kA61_Ev0Pvwhu_7_1TD9fEeLRgoURa0J?usp=sharing

**虽然笔者写这篇文章的初衷是复习自己所学的知识，但如果你能给我一个小小的赞，笔者会十分开心的！**



