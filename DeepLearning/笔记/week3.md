# 梯度爆炸

对于具有非常多隐藏层的结构，如果参数矩阵W的值随着层数的增加，而是指数级别的增长，则梯度会爆炸



## 慎重的初始化

方法1：对第L层神经元，初始化为：随机数乘以$\frac{1}{n}$的平方根 ，n为该层的输入特征数（即上一层的输出维度）；如果激活函数是ReLu，建议改为$\frac{2}{n}$ 的平方根。其他不同的激活函数，也相应有不一样的常用初始化。

同时也可以通过设置一个超参数的方式，找到这个初始化应该乘以的数字



# 检查反向传播

==对于首尾拼接向量，这一部分还是有点疑惑，不知道是不是这样处理的==

通过比较笨的方法，即$f^{\prime}(x)=\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}$ 计算一个梯度出来。然后再计算通过导数公式算出来的值，看一下两个的误差是多少，保证导数公式正确



对于整个神经网络，通常的过程如下
$$
W^{[1]},b^{[1]},W^{[2]},b^{[2]}\cdots W^{[L]},b^{[L]} 每个都转换为一维向量，然后首尾拼接成一个向量\theta
\\
dW^{[1]},db^{[1]},dW^{[2]},db^{[2]}\cdots dW^{[L]},db^{[L]} 每个都转换为一维向量，然后首尾拼接成一个向量d\theta
$$
之后，对于$\theta ,d\theta$中的每个分量进行计算

![image-20230115133646126](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20230115133646126.png)

然后计算$d\theta_{approx}和d\theta$ 的欧几里得距离，如果在1e-5~1e-7，可认为正常



==梯度检验不能和dropout一起使用==





# mini-batch

mini-batch的思想是：将整个数据集分成多个小的mini-batch，即划分为不同的小训练集。然后对于每个小的训练集都进行一次梯度下降计算，这样也就说：当遍历完整个数据集的时候，会进行mini-batch数量这么多次的梯度下降。这样可以得到，同样的遍历那么多样本，但更多次的梯度下降



通常mini-batch的梯度下降画出来是有点特别的，并不是单调下降的，但整体趋势是下降的

![image-20230115135718553](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20230115135718553.png)

mini-batch的大小越接近整个数据集的大小，梯度下降的噪声越小。

如果mini-batch大小为1，称为随机梯度下降

==一般设为2~整个数据集大小==

根据经验：

1、数据集小于2000，直接batch就行

2、其他情况，mini-batch设置大小为：64，128，256，512这些2的倍数

3、保证mini-batch能够放进cpu、gpu 的内存中

4、==mini-batch也是一个超参数==





# EMA

修正的EMA

目的使得一开始的时候，EMA后的值不至于太小
$$
V_{t}=\frac{(1-\beta)\theta_{t}+\beta V_{t-1}}{1-\beta^{t}}
$$




# 梯度更新方法

## Momentum



## RMSprop





## Adam





# 学习率衰减

**学习率衰减公式**
$$
\alpha=\frac{1}{1+decay\_rate*epoch}\alpha_{0}\\
\alpha=0.95^{epoch}*\alpha_{0}\\
\alpha=\frac{k}{\sqrt{epoch}}\alpha_{0}\\
$$
decay_rate为超参数，$\alpha_{0}$为初始学习率
