本文基于：https://www.cnblogs.com/peghoty/p/3857839.html



# 模型目标

CBOW实现的是：1、给定一个词的上下文，预测这个词是什么；2、训练词向量



# 模型架构

模型由三部分组成：

![image-20221001153246536](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20221001153246536.png)



## Input layer

已知我们有一个训练数据**(Context(w),w)**，**Context(w)表示词w的上下文**

我们将Context(w)中的每个单词的词向量作为输入，输给Projection Layer

注意：这里输入的是每个词的词向量，但我们一般不能直接得到词向量，所以其实这一步之前是有其他的处理步骤的，详细内容见fasttext文章



##  Projection Layer

投影层的任务就是将从Input Layer得到的数据进行转换。这里做的是：将Context(w)中的每个词向量相互累加，得到一个全新的向量$X_{w}$ 

![image-20221001155029549](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20221001155029549.png)

$X_{W}$ 的计算方式有很多种，不一定是相互累加



## Output Layer

这里就是重中之重了。Output Layer的作用是：输出我们最终的结果

CBOW在Output Layer中使用的模型是：**Hierarchical softmax**，一个树模型的结构



### 工作原理：Hierarchical softmax

在进行复杂的数学推导之前，先讲一下这个树结构是怎么工作的。

1、首先要明确的是，这棵树是由数据中所有词共同组成的树，其中每个叶子结点都代表了**一个词**。换句话说，数据中的每个词都会在这棵树的叶子结点中出现

2、这棵树是一棵**哈夫曼树**，一般根据词出现在数据中的频率构造的

3、非叶子结点可以称作是参数结点，这是我们想要训练的一个参数

4、怎么做出预测？假设我们有一个样本，表示一个词的上下文，我们通过Input Layer和Projection Layer对样本进行了转换。之后将这个样本传入哈夫曼树，我们可以计算树中每个词的预测值。根据预测值，就能得到最后的结果。这里有个技巧，可以利用深度优先策略加速预测过程，因为子结点的值会小于父节点的值。

![image-20221001155433898](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20221001155433898.png)







### 模型训练

#### 规定符号

![image-20221001200430772](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20221001200430772.png)

6、C 表示词典

**注释**

对于$\theta$ ，其实就是经过非叶子节点时，样本（我们输入的上下文在hidden层后转换成的向量$X_{w}$ ）要乘的一个参数。例如$\theta^{w}_{j}$ 表示：在到词w对应的叶子节点的路径上， 第j个非叶子结点所带有的参数，这个参数就是用作算概率值的。



#### 小结

对于词典$D$ 中的任意词$w$ ，Huffman树中必存在一条从根节点到词$w$ 对应结点的路径 $p^{w}$ （且这条路径是唯一的）。路径 $p^{w}$ 上存在$l^{w}-1$ 个分支，将每个分支看作一次二分类，每一次分类就产生一个概率，将这些概率乘起来，就是所需的 $p(w|Context(w))$ 



#### 目标函数

我们由上述小结可得
$$
p(w|Context(w))=\prod_{j=2}^{l^{w}}p(d_{j}^{w}|X_{w},\theta^{w}_{j-1})
$$
也就是说： $p(w|Context(w))$ 等于路径上所有非叶子节点预测的概率值的累乘。而每个非叶子节点预测的概率值。



而每个非叶子节点预测的概率值，是一个logistics二分类问题，所以有：
$$
p(d_{j}^{w}|X_{w},\theta^{w}_{j-1})=\begin {cases}
\sigma(X^{T}_{W}\theta^{w}_{j-1})  & {d_{j}^{w}=0}  \\
1-\sigma(X^{T}_{W}\theta^{w}_{j-1})&{d_{j}^{w}=1}
\end{cases}
$$
其实就是一个logistic分类，$\sigma(X^{T}_{W}\theta^{w}_{j-1})$ 表示$d_{j}^{w}=0$的概率 。注意，这里计算的概率值表示的是这个结点之后往右（编码为0）还是往左（编码为1）走的概率。

将上述分段函数整体表达为：
$$
p(d_{j}^{w}|X_{w},\theta^{w}_{j-1})=[\sigma(X^{T}_{W}\theta^{w}_{j-1})]^{1-d^{w}_{j}}\cdot[1-\sigma(X^{T}_{W}\theta^{w}_{j-1})]^{d^{w}_{j}}
$$


这样我们可以得到$p(w|Context(w))$ 的整体表达式
$$
p(w|Context(w))=\prod_{j=2}^{l^w}{[\sigma (X^{T}_{W}\theta^{w}_{j-1})]^{1-d_{j}^{w}}*[1-\sigma (X^{T}_{W}\theta^{w}_{j-1})]}
$$


我们的目标是使得词典C中的每个词的$p(w|Context(w))$都尽可能的大，因为这代表我们对于真实值的预测越准确。因此我们将$p(w|Context(w))$ 的整体表达式代入对数似然函数，可得我们的目标函数
$$
\ell =\sum_{w \in C}\log \prod_{j=2}^{l^w}{[\sigma (X^{T}_{W}\theta^{w}_{j-1})]^{1-d_{j}^{w}}*[1-\sigma (X^{T}_{W}\theta^{w}_{j-1})]}\\
=\sum _{w \in C}\sum_{j=2}^{l^{w}}{(1-d^{w}_{j})*\log[\sigma (X^{T}_{W}\theta^{w}_{j-1})]+d^{w}_{j}*log[1-\sigma(X^{T}_{W}\theta^{w}_{j-1}) ]}
$$
其中，单个节点（树中的非叶子结点）的目标函数如下
$$
\ell (w,j)=(1-d^{w}_{j})*\log[\sigma (X^{T}_{W}\theta^{w}_{j-1})]+d^{w}_{j}*log[1-\sigma(X^{T}_{W}\theta^{w}_{j-1}) ]
$$


**解释一下**
1、$\ell$ 是所有叶子节点的$\ell(w,j)$ 的总和

2、需要寻找的参数有$X_{w},\theta$ ，$X^{T}_{w}$是$X_{w}$ 的转置

3、单个样本目标函数值：这个值是根据构造的哈夫曼树来的。这个值等于从根节点到这个叶子节点的路径上，所有节点的logistics二分类概率值累乘。让样本的目标函数值越大越好。

4、整个模型的目标函数等于：所有样本的目标函数值的累加

注意：每个非叶子结点上都是有两个值的，一个是向右走的值，一个是向左走的值。CBOW规定向右编码为0，为正样本。像右走时，即$d^{w}_{j}=0$ ，该结点值等于：$\ell (w,j)=\log[\sigma (X^{T}_{W}\theta^{w}_{j-1})]$  ；向左走,即$d^{w}_{j}=1$，结点值等于 $ \ell =log[1-\sigma(X^{T}_{W}\theta^{w}_{j-1}) ]$ 。其实就是一个logistics的判断。



#### 目标，随机梯度上升法

现在我们的目标是：每个样本传入，到他所属的叶子节点时，叶子节点的预测值越大越好，最好能接近1。也就是说，我们要使得目标函数越大越好。

问：那怎么样才能使目标函数越大越好呢？

答：我们可以利用随机梯度上升法，对参数$X_{w},\theta$ 进行调整



**关于$\theta_{j-1}^{w}$ 的梯度计算**

![image-20221001202840084](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20221001202840084.png)



**关于$X_{w}$ 的梯度计算**

![image-20221001202923560](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20221001202923560.png)



更新$X_{w}$如下：$\tilde w\in Context(w)$ 

![image-20221001203045742](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20221001203045742.png)

即把梯度作用在组成$X_{w}$ 的每个词上，达到更新$X_{w}$ 的目的，同时还能完成训练词向量的任务。注意：每传入一个样本，$X_{w}$ 和$X_{w}$ 中每个词的词向量都只更新一次。更新的梯度是路径上所有非叶子结点的梯度和



**总结**

每传入一个样本，这个样本的形式是二元对$(Context(w),w)$ ,我们都根据梯度，进行依次参数的更新。更新的是从根节点到词w的路径上经过的非叶子节点的$\theta$ 和更新一次$X_{w}$ 和$X_{w}$ 中每个词的词向量。



#### 伪码

![image-20221001204727247](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20221001204727247.png)





# 总结

至此，我们了解了 CBOW的工作流程以及其中的一些数学推导。

CBOW希望实现的是：基于上下文预测中间词，而词向量只是他的一个副产物。通过Hierarchical softmax，CBOW能够更快速的处理词典很大的情况。