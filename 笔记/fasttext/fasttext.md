# 介绍

**fastText是**[Facebook](https://en.wikipedia.org/wiki/Facebook)的 AI 研究 (FAIR) 实验室创建的用于学习[词嵌入](https://en.wikipedia.org/wiki/Word_embedding)和文本分类的模型。人们可以通过该模型完成无监督学习或监督学习。同时Facebook 已为 294 种语言提供预训练模型。它的有点在于：不仅能够获得媲美深度学习模型的精度，而且得益于其简单的结构，它能够有着更快的训练速度。在标准的多核CPU上， 能够训练10亿词级别语料库的词向量在10分钟之内，能够分类有着30万多类别的50多万句子在1分钟之内。



虽然fasttext如此出色，但是其实它在模型上并没有什么大的创新。fasttext是在基于CBOW模型的基础上，引入了词内的n-gram，使得模型相比于CBOW有着一定的改进。





# 模型的结构

![image-20220930173706273](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220930173706273.png)

注意：图中省略了Input Layer



## Input 和 Hidden Layer

**为了便于理解fasttext的Input Layer和Hidden Layer，下面先从一个例子讲起。**

假设我们现在有一个数据集，这个数据集只有一个文本：“A B C D. ” 根据数据构造有词典D，其中单词A-D的索引为1-4。假设我们用最简单的**one-hot** 对A、B、C、D进行编码，则A可以表示为向量:[1,0,0,0]

Input和Hidden Layer的工作原理如下图

![image-20221002182459652](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20221002182459652.png)

1、输入每个词的向量表示，如A:[1,0,0,0]

2、通过矩阵W，将每个词的向量转换为特点维度的向量，一般这个向量被称作**词向量**（图中1 x h的矩阵）。参数h是一个超参数，是我们可以自主设置的

3、在Hidden层，将所有的词向量加总算平均， 得到的向量传给Output层。这个向量$X_{w}$ 可以被认为是文本的向量表示



fasttext的Input Layer和Hidden Layer大致上与上述过程一样。但在fasttext中，因为加入了词内的n-gram，所以输入的不单单只是文本中每个词的向量，还有词的n-gram向量。

问：那词的n-gram向量我们是怎么计算的呢？

答：对于一个词的n-gram向量

1、先根据所有词的“所有n-gram可能“形成一个n-gram词典G

2、与表示一个词类似，我们也可以利用one-hot的形式，表示这个词的所有n-gram可能

3、将这个词的所有n-gram可能的向量加总求和，即为这个词的n-gram向量表示

当然，我们还可以有词与词之间的n-gram（只是这个计算方式和词内的不一样，可以简单的表示为两个词的向量表示相加）



如果你之前是有了解过CBOW的，那你一定会记得CBOW通过计算梯度的方法，能够更新词向量。在fasttext中是一样的。通过反向传播，计算$X_{w}$ 的梯度，来更新矩阵W和每个词的词向量。





## Output

利用hidden层输出的数据，在output层进行计算。在output层利用的模型是Hierarchical softmax

为了理解怎么样进行预测，有必要先理解Hierarchical softmax



### Hierarchical softmax

因为fasttext是基于CBOW实现的，所以同样的也用到了Hierarchical softmax技术，详细内容可见CBOW文章



为什么能够加快train和test的训练速度？

答：很简单，我们将更常出现的类摆在树的上层，也就是说训练更常出现的类的时候，经过的非叶子结点（参数结点）更少，需要进行更新的参数更少，计算量自然就少了

而至于为什么是$O(h\log_{2}k)$ ，猜想：最坏的情况，每个类出现的次数是一样的，也就是每个类都要算那么多。此时构建的哈夫曼树就是一个平衡二叉树，高度就是$\log_{2}k$ 。h表示的trian训练集中有多少行样本。每个样本的训练都要经过大约$\log_{2}k$个（这里没有仔细算）的参数节点，也就是要经过这么多次的计算（参数更新）。所以时间复杂度是$O(h\log_{2}k)$。

而原来的时候，没有哈夫曼树时，对于每个样本，也就是每个类的参数都要进行计算。所以就是一个样本要算k次参数更新。而哈夫曼树的是，只会更新这个样本走过的路径的点，不会更新与他无关的点。



## summary

与CBOW相似，计算的是预测为样本是实际类的概率值。这个值的计算方法和CBOW中Hierarchical softmax的方法一样，基于哈夫曼树计算。也就是说类的值的计算与CBOW中每个词的计算相同，是许多个logistics合成的，形成softmax的效果



根据fasttext论文，我们的目标函数如下：是一个负对数似然（常用于softmax function）

![image-20220930174523863](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220930174523863.png)

**注释**

1、$f()$ ：sofmax function（因为我们选用softmax 作为output）

2、B和A为权重矩阵

3、$y_{n}$ 为第n个文本的标签，一般为一个向量表示，$x_{n}$ 为第n个文本的向量表示

4、N为总共的文本数量，即数据量



但实际上，利用Hierarchical softmax的话，上述目标函数是不对的，虽然论文中这样写了。实际的目标函数应该是CBOW中的Hierarchical softmax的目标函数。

之后采用梯度下降的方法求目标函数最小值，得到参数



# 优化

fasttext的特点在于：提出了词内的n-gram。根据词内的n-gram，我们能够对于没有在样本中出现过的词也能较好的表示，提升了模型的一个泛化能力。



## 词内n-gram

因为内容较多，但比较容易理解，所以给出资料，不详细叙述

资料：https://adityaroc.medium.com/understanding-fasttext-an-embedding-to-look-forward-to-3ee9aa08787





# 代码

官方说明文档：https://fasttext.cc/docs/en/python-module.html

colab:https://colab.research.google.com/drive/1k145zrzroRT7JdPxWoABaRvc13kxcBfN#scrollTo=myVxcPLyMFPt

[CSDN博客](https://blog.csdn.net/weixin_45707277/article/details/122794848?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166453914016800184113361%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166453914016800184113361&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-122794848-null-null.142^v51^control,201^v3^control_1&utm_term=fasttext&spm=1018.2226.3001.4187)



1、fasttext能够使用的数据格式如下 ：label标识分类，可以有多个label同时存在。label和label，label和文本，用Tab隔开

```python
"__label__1	__label__2 文本内容 "
```

2、主要函数

```python
#监督学习
fasttext.train_supervised(txt格式的数据)
#无监督学习
model = fasttext.train_unsupervised(txt格式的数据, model='')
#自动超参数寻优
model = fasttext.train_supervised(txt格式的数据,autotuneValidationFile=验证集(txt格式), autotuneDuration=600)#使用autotuneDuration参数可以控制随机搜索的时间
#save and import model
model.save_model('model_name.bin')
model = fasttext.load_model("路径/model_name.bin")
```

3、model.predict(["文本内容"], k=3,threshold=0.5)，预测文本内容的分类，k控制返回最高排名的多少类



## 超参数

![image-20220929163937456](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220929163937456.png)

几个重要参数

| 超参数     | 范围                   | 影响                                                         |
| ---------- | ---------------------- | ------------------------------------------------------------ |
| dim        | int，一般取100-300     | 控制词向量维数，影响训练速度                                 |
| epoch      | int                    | 迭代次数，越多越容易过拟合                                   |
| minn       | int                    | 词内n-gram最小长度                                           |
| maxn       | int                    | 词内n-gram最大长度（如果想保持一个长度的词内n-gram，则minn=maxn） |
| ws         | int                    | 要考虑中心词的前后多少个词skip-gram和CBOW不一样）            |
| wordNgrams | int                    | 每次取词取多少个（skip-gram和CBOW不一样），等于词间的n-gram的n。经验表明一般在3-6之间 |
| loss       | {ns, hs, softmax, ova} | ova专用于多分类，使得模型等于训练多个二分类模型，对每个类都是独立的预测 |

wordNgrams和ws详细解释见[链接](https://stackoverflow.com/questions/57507056/difference-between-max-length-of-word-ngrams-and-size-of-context-window)



























