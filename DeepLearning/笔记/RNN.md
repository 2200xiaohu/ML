# RNN简单结构

1、每一层都会考虑前一层的情况，相当于每一层都会有两个输入，一个是前一层的a，一个是新输入的词

2、计算每一层的 a 有三个参数$W_{aa},W_{ax},b_{a}$ 
$$
a^{<t>}=g(W_{aa}a^{t-1}+W_{ax}x^{<t>}+b_{a})
$$
![image-20230301111252615](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230301111252615.png)





# GRU

![image-20230307111032933](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230307111032933.png)



# LSTM

![image-20230307111111469](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230307111111469.png)



# Word Embedding

例如：一个1 X 512 的向量

可以理解为：

1、每一维都表示一个特征，数值表示该词这个特征的显著性

例：维度1如果表示Age，对于词MAN，那该向量的第一维取值应该约等于1



## t-SNE

使用t-SNE算法，将一个高维的向量表征成2维的，便于可视化，计算相似度，进行分组





# Word2Vec  （word embedding）

## skip-gram

每次在句子中选择一个词作为content，然后在该词的前后一个范围内，选择不同的词作为目标target

即中间词预测前后文中的任意词

等于说：content作为输入，target作为标签

输入到一个简单的softmax函数中，完成训练



存在有问题：每次计算softmax时，分母难以计算

解决办法：负采样、hierarchical softmax



## CBOW

每次在句子中选择一个词作为target，然后在该词的前后文（一定长度）作为content

即：前后文预测中间词



## 负采样

例如在skip-gram中的负采样



我们要根据content预测不同的词，首先构造一个数据集

数据集由两部分组成：1、选取的词就是出现在content后面的；2、随机在词典中抽取词，假设其为预测的target，这部分就是负样本

构造的数据集如下图。正确采样的target为1，其他的为0

这样也就转换为了一个二分类问题：是否是真正跟在content后面的词？而不是找到跟在后面的词是什么

效果：没计算一个content的时候，不再需要计算所有样本组成的softmax，而是变成了关于k+1个样本的二分类问题

![image-20230308104536478](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230308104536478.png)

如何选取应该负采样多少个？

数据集较少的时候，建议选取k=5 - 20 个

较多的时候，k = 2 - 5



如何采样负样本？

没有标准的方法