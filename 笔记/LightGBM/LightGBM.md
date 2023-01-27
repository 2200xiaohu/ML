https://towardsdatascience.com/what-makes-lightgbm-lightning-fast-a27cf0d9785e



# 主要改进

- 基于Histogram的决策树算法
- 单边梯度采样 Gradient-based One-Side Sampling(GOSS)
- 互斥特征捆绑 Exclusive Feature Bundling(EFB)
- 带深度限制的Leaf-wise的叶子生长策略
- 直接支持类别特征(Categorical Feature)







# 基于Histogram的决策树算法

其实就是加权直方图，原理上是一样的，但引入了一个新的方法

直方图做差

一个结点的加权直方图可以由其父结点的减去其兄弟节点 ？？？？？？？？







# 带深度限制的Leaf-wise的叶子生长策略

XGboost采用的是按层生长，宽度优先（**Level**-wise）的策略，每次能够同时生成同一层的所有结点

但实际上，同一层的有些结点的信息增益是比较小的，没有必要进行生成，造成了不必要的浪费

所以LightGBM采用“带深度限制的**Leaf**-wise”，策略是：类似于XGboost的策略，但不再是一次性生成同一层的所有结点，而是选择一个最优的进行分裂。产生的效果是：能有更高的精度，但树的深度会更深，更加容易过拟合。所以又附加了一个限制，限制一个最深的深度





# 单边梯度采样 Gradient-based One-Side Sampling(GOSS)

LightGBM模型的训练是基于样本的梯度的，梯度大的样本表示模型学习的还不太好，是需要更加关注的。而对于梯度小的样本，我们甚至可以战略性的放弃，以提高模型的训练速度。所以我们可以基于每次week learner产生的梯度，重新对样本进行采样。

具体实现：

1、根据梯度的绝对值进行降序排序

2、选择top a%的样本

3、从剩下的样本中，随机抽样b%的样本

4、调整小梯度样本的权值（应该是体现在梯度上面），乘一个常数：1-a/b

5、将抽样的样本和调整的权值集合，传入weak learner





# 互斥特征捆绑 Exclusive Feature Bundling(EFB)

将一些相互冲突的特征合并成一个特征。最常见的是one-hot encoding形式的特征，即这些特征不同时取零

我们可以通过添加偏移量，对这些特征进行加总，生成一个新的特征，达到类似于降为的目的







# 超参数

## 调参技巧

https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

### 防止过拟合

1、设置num_leaves（树最大叶子数）小于$2^{max\_depth}$ ，如果是等于，则与XGBoost的depth-wise策略一样，容易造成过拟合

2、min_data_in_leaf：能够做叶子结点的样本数量，防止过拟合，但数值过大容易造成欠拟合。对于大数据，三位数或者四位数就够了。如果一个结点分裂后的两个叶子中，有一个的样本数量不符合这个，则这个特征不考虑这样分裂

3、max_depth防止树过深



### 加速训练速度

1、通过设置超惨，减少模型的复杂度

减少：max_depth,num_leaves

增加：min_gain_to_split（能够分裂的最小增益值）、min_data_in_leaf、min_sum_hessian_in_leaf（能够分裂的残差的二次导数的和最小值）

2、减少weak learner 的数量

num_iterations：weak learner 数量

3、设置early_stopping_round：设置阈值，如果之后效果没提升，则结束

4、设置feature_pre_filter=True：当设置了min_data_in_leaf，在传入数据前先检查是否有特征不符合这个的，则预先将这个特征删掉，不考虑

5、减少max_bin & max_bin_by_feature 

max_bin:设置加权分类法的分箱数目，max_bin_by_feature：对每个分别特征设置max_bin

6、min_data_in_bin：设置每个箱的最小样本数目

7、feature_fraction：随机抽取百分之多少的特征用作新的树的特征（一般就为1比较好）

8、max_cat_threshold：对分类特征进行处理，越大，尝试越多分裂值

9、bagging_freq 和bagging_fraction一起使用：{"bagging_freq": 5, "bagging_fraction": 0.75}表示，每5个weak learner重新采样0.75的样本





## 超参数总结

所有超参数总结

https://lightgbm.readthedocs.io/en/latest/Parameters.html

### LGBMRegressor（sklearn API）

https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor

https://lightgbm.readthedocs.io/en/latest/Parameters.html

| 超参数                | 范围                           | 作用                                                         |
| --------------------- | ------------------------------ | ------------------------------------------------------------ |
| **boosting_type**     | gbdt,dart,goss,rf              | ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest. |
| **num_leaves**        | int,default=31                 | 最大叶子结点数量                                             |
| **max_depth**         | int,default=-1                 | 最大深度，小于等于0表示没有限制                              |
| **learning_rate**     | float                          | 学习率。可以通过将fit函数传入callbacks，实现在train时候调整学习率，则此参数会被忽略 |
| **n_estimators**      | int                            | weak learner 数量                                            |
| **subsample_for_bin** | int ,default=200000            | 用于产生分箱的数量                                           |
| **objective**         |                                | 目标函数：Default: ‘regression’ for LGBMRegressor, ‘binary’ or ‘multiclass’ for LGBMClassifier, ‘lambdarank’ for LGBMRanker. |
| **class_weight**      | 'balanced' or None, （可选的） | 只适合多分类任务，用于设置类的权重。如果是二分类，用的超参：is_unbalance，scale_pos_weight |
| **min_split_gain**    | float                          | 能够分裂的最小增益值                                         |
| **min_child_weight**  | float                          | 孩子节点（叶子节点）最小残差二次导数的和                     |
| **min_child_samples** | int                            | 孩子节点（叶子节点）最小个数                                 |
| **subsample**         | float                          | 取数据的百分之几作为训练集                                   |
| **subsample_freq**    | int                            | 多少个weak learner重新取样， <=0 means 不进行变动            |
| **colsample_bytree**  | float                          | 每生成一棵树随机采样多少列作为特征                           |
| **reg_alpha**         | float                          | L1正则化                                                     |
| **reg_lambda**        | float                          | L2正则化                                                     |
| **random_state**      | int                            | 随机数种子                                                   |
| **n_jobs**            | int                            | 线程数                                                       |
| **importance_type**   | split，gain                    | 特征重要性计算方式，split统计特征使用次数，gain统计特征的贡献 |
|                       |                                |                                                              |









https://koreapy.tistory.com/m/758

https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_tree.html
