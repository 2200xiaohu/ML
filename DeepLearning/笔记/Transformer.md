# Attention

## self-attention

## 基本结构

1、每个input vector对应有一个输出
但需要一次性输入所有的input vector，才能学习到相互之间的关系

2、self-attention可以堆叠



![image-20230317152100411](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20230317152100411.png)



### 输入和输出

输出可以同时被计算出来，不需要一个个依次计算

![image-20230317152138646](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20230317152138646.png)





## 计算过程



### 前提

**假设现在需要对输入$a^{1}$ 通过self-attention得到对应的output  $b^{1}$**



### Part1：计算attention_score

attention_score是一个中间量，表示的是两个单词之间的关联性、

计算方法有很多种，常用有**两种**，如下图。其中左边，两个向量点乘更为常用

![image-20230317152320155](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20230317152320155.png)



放在整个模型中，attention_score的计算如下：

1、这里选择点乘的方法计算

2、获取$a^{1}$ 以外的其他输入的$k$ 值，然后点乘$a^{1}$ 的 q值

3、得到的$\alpha$ 为attention_score



数学过程
$$
k^{i} = W^{k}*a^{2}\\
\alpha_{1,2} = q^{1} * k^{i}
$$
**注意**：图中这里不完全，应该添加上计算 $q^{1} * k^{1}$ ，即自身也要算一次

![image-20230317152615463](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20230317152615463.png)





### Part2 ：激活层处理

将得到的所有attention_score 传入一个激活层，这个激活层可以是softmax也可以是其他的激活函数，例如relu等等

![image-20230317153331785](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20230317153331785.png)





### Part 3： 最终输出

1、将刚刚得到的，通过激活层之后的attention_score，图中符号为$\alpha_{1,1}^{'}$ ，与其本身对应的value，即$v^{1}$ 相乘

2、将所有输入都进行一样的操作，累加，得到最终的输出 $b^{1}$

![image-20230317153356451](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20230317153356451.png)



### 写成矩阵乘法

I 代表输入

![image-20230318100221753](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318100221753.png)



## 如何理解q，k，v？

**个人理解**

**Q**：你在浏览器中输入了一个问题
**K**：浏览器返回的不同网页（回答）

**attention_score**：由Q和K计算得到，代表我们对每个网页的质量判断，判断我们觉得哪个网页应该更容易找到问题的解答。也可以说代表关联性

**V**：每个网页中的内容

**b**：由attention score和V计算得到，代表我们最后得到最终答案。可以理解为，在浏览了这么多网页之后，根据对每个网页重要性的判断，综合了所有网页的内容得到的最终解答



## Multi head self-attention

简单来说就是：

1、一个输入，能产生多个q，k，v

所以，有多少个head，就有多少个不同的q，k，v

对应有，多个$W^{q},W^{k},W^{v}$

2、在计算attention_score的时候，是第一个head的q，k，v之间进行计算的。不同的head之间不能交叉

3、在得到多个head的b之后，通过一个矩阵，得到最终的输出。

所以Multi head self-attention 依然是一个输入，一个输出



![image-20230318102312497](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318102312497.png)



![image-20230318102544774](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318102544774.png)



## Guide Attention

控制Attention 的 训练过程，例如强迫Attention是从左向右计算结果的



# Transformer

1、是一个sequence2sequence的model

2、模型的输出可以是由模型自己的决定的，例如在语音识别和翻译中





## Encoding

由很多个Block组成，每个block都可以看成是一个小的neural network，里面由很多层

![image-20230318112525806](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318112525806.png)

在Transformer里面的Encoding中，每个Block结构如下

1、先做一个self-attention，注意这里加上了一个残差连接

2、对self-attention的输出做一个**Layer norm**（注意，这里不是Batch Norm）。Layer norm的计算L：对每一列分别计算该列的mean和方差，然后norm，公式如图，其中有一个小的错误，等号右边的分子，xi不要上标

3、在之后，做一个FC，同样也是带有残差连接额Layer norm 的

4、这样就得到了这个Block的输出

注意：Block中的内容不一定要这样设计，是可以有改进的

![image-20230318112456971](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318112456971.png)

对应在整个Transformer中，每个Block的构成可以表示成下图形式

1、Multi-Head Attention ： 就是Block中的self-attention

2、Add & Norm ： Block中self-attention后的残差连接和Layer norm

3、Feed Forward ：Block中的FC层 

![image-20230318113356210](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318113356210.png)



## Decoder

一般有两种Decoder

1、Autoregressive：循环产生输出，每次循环都只有一个output

2、Non-Autoregressive ：一次性产生所有output



**Decoder 结构**

相比于Encoder，其实就是改了一个Masked Multi Head Attention和多了一个连接Decoder的部分

![image-20230318120654539](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318120654539.png)





### Autoregressive

Autoregressive的Decoder：相当于一个RNN单元，即自己的输出会当成下一次的输入

所以，这种Autoregressive下的Decoder的输出，是一个个产生的，每次循环都产生一个最终输出



什么时候停止呢？

答：在当Decoder的输出是我们设定的End符号时就停止。所以我们需要在Decoder的输出向量中，多加一个位置，表示End



### Non-Autoregressive

一次性产生所有的Output

但如果在无法确定Output长度的时候怎么办呢？

答：两种做法

1、将Decoder的输入另外fed给一个classifier，判断Output的长度

2、将Output中End符号之后的输出全部不要，相当于在End位置截断了



优点：

1、并行计算

2、能够人为的控制输出长度





### Masked Multi Head Attention

Maksed Multi Head Attention的改变是：

1、在原本的Multi Head Attention中，每个输出b，都是根据所有的输入计算的

2、现在改为对于$a^{i}$ 的输出 $b^{i}$ ，只能根据从$a^{1}$  到 $a^{i-1}$ 计算。相当于只考虑在当前输入之前的输入数据

3、Masked Multi Head Attention的输入其实是Decoder的之前的所有输出，所有每次的计算，其实就只输出一个向量，即当前输入对应的输出



为什么需要masked？

答：因为Autoregressive下Decoder的输出是一个个产生的，并不是一次性就产生整个输出。放在self-attention中，就是Masked Multi Head Attention了，每次都只考虑之前的输入，因为后面的照道理来说是不可见的输入



注意：Masked Multi Head Attention的输入应该是逐渐变长的，所以需要记忆之前输入的数据。同时，在Transformer中使用了一个Trick，对Encoding 输入正确答案，而不是我们预测的结果



## Cross attention（连接Decoder和Encoder）

在模型中的位置

![image-20230318121123376](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318121123376.png)



结构如下

![image-20230318121002572](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318121002572.png)

计算过程：

1、Decoder的Masked Multi Head Attention和Encoder的输出 fed 给在**Decoder**中的**Multi Head Attention**

注意，这里Encoder的输出的符号是$a$

2、做Multi Head Attention 的**子操作**（见注意点），得到output v

3、每当Masked Multi Head Attention输出一个向量，就重复一次上述步骤

注意点：

1、子操作：在Decoder的Masked Multi Head Attention的输出，只需要提供q，不需要提供v和k。因为这里仅仅只是计算Decoder 的Masked Multi Head Attention的输出与Encoder的关系



第二次循环的计算过程

![image-20230318122418710](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318122418710.png)



一些改进：

刚刚上面介绍的是每次都是拿Encoder的最后一个Block的输出作为输入，但其实可以有改进的

可以有不同的连接方式，例如对应位置Block的Decoder和Encoder输出进行连接，即Encoder的第一个Block输给Decoder的第一个Block，依次类推

![image-20230318122906809](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318122906809.png)



## Positional Encoding

**作用**：给输入元素添加位置信息

方法：非常简单，就是简单的在输入向量上，加上一个能够表示位置信息的向量$e^{i}$ 

这个向量$e^{i}$ 有很多种方法构建出来，但现在是没有一个标准的。

常用有：

1、sin-cos方法

2、通过网络学习出来



![image-20230318103116626](C:\Users\nigel\AppData\Roaming\Typora\typora-user-images\image-20230318103116626.png)







# 问题：Decoder每次的输入是怎么样的

## Teacher Forcing

Decoder的输入是到当今循环下，对应长度的正确答案



问题：那如果是在Test的时候，没有正确答案，那Decoder的输入又是什么？这个问题叫做：exposure bias

答：可能的解决办法：在训练的时候，加一下错误的输入进去



# 扩展知识

1. Beam Search
2. Guide Attention
3. 当你不知道怎么选择loss的时候，用reinforcement learning（RL）硬试一下

