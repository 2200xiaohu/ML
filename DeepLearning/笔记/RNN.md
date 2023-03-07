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