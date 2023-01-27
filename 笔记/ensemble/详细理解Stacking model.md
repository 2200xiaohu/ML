 # 详细理解Stacking model

如果你得到了10个不一样的model，并且每个model都各有千秋，这个时候你该怎么选？想必你一定是很为难吧，但通过集成方法，你可以轻松的将10个model合成为1个预测更精确的model。今天要介绍的就是众多集成方法里面的"Stacking"



## 什么是Stacking？

Leo Breiman 以他在分类和回归树以及随机森林方面的工作而闻名，他在 1996 年关于*堆叠回归的论文* (Breiman [1996 ](https://bradleyboehmke.github.io/HOML/stacking.html#ref-breiman1996stacked)[b](https://bradleyboehmke.github.io/HOML/stacking.html#ref-breiman1996stacked) )中将堆叠形式化。尽管这个想法起源于（Wolpert [1992](https://bradleyboehmke.github.io/HOML/stacking.html#ref-stacked-wolpert-1992)），名为“Stacked Generalizations”，但使用内部 k-fold CV 的现代堆叠形式是 Breiman 的贡献。

（Wolpert的文章获取：https://www.researchgate.net/publication/222467943_Stacked_Generalization）

然而，直到 2007 年，堆叠的理论背景才被开发出来，并且当算法采用了更酷的名称Super Learner （Van der Laan、Polley 和 Hubbard [2007](https://bradleyboehmke.github.io/HOML/stacking.html#ref-van2007super)）。此外，作者说明超级学习者将学习基础学习者预测的最佳组合，并且通常表现得与构成堆叠集成的任何单个模型一样好或更好。直到此时，堆叠工作的数学原因尚不清楚，堆叠被认为是一门黑色艺术。

**<u>模型堆叠（Stacking）</u>**是一种有效的集成方法，其中使用各种机器学习算法生成的预测被用作第二层学习算法的输入。该第二层算法经过训练，可以优化组合模型预测以形成一组新的预测。例如，当线性回归用作第二层建模时，它通过最小化最小二乘误差来估计这些权重。但是，第二层建模不仅限于线性模型；预测变量之间的关系可能更复杂，从而为采用其他机器学习算法打开了大门。

![image-20220828170028207](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220828170028207.png)

![stackedapproach](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/stackedapproach.png)

一般来说，Stacking由两层组成就够了。但完全可以由多层组成，其中一些层可以用作噪声的处理等等。其实不难发现，多层的Stacking与Deep learning是有点相似的。

All in all ，Stacking一般由两层组成。**第一层**：表现出色的基本模型；**第二层**：将第一层模型们的输出作为训练集得到的模型。第二层模型又被称作”meta-model 

“，关键作用在于将第一层的所有模型的结果整合起来，进行输出。也就是说，第二层模型将第一层模型的输出作为特征进行训练。

![1_0qQTUDfImZYQBsyn9F6dpw](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/1_0qQTUDfImZYQBsyn9F6dpw.png)



## 不同种类的Stacking

在Stacking的实际应用中，有两者Stacking的方法：无cv（交叉验证）和有cv的方法。有cv的方法是无cv方法的一个改进，目的是避免第二层meta-model过拟合的集成第一层的模型。下面，先从最简单的”无cv“ Stacking开始



## 无Cross-Validation Stacking

无cv Stacking就是Stacking最原始的方法，搞懂它，你就可以说是搞懂了Stacking的原理了。受限于笔者的水平，笔者无法从数学上去解释Stacking的原理以及为什么能这样做（直到现在世界的顶尖学者也无法解释其数学原理）。笔者在阅读了Saso Džeroski,Bernard Ženko于2004发表在Machine Learning期刊上的《Is Combining Classifiers with Stacking Better than Selecting the Best One》后，从网上搜集了一些资料（link都在有关文献中），斗胆的说已经大致理解了Stacking的原理。下面，笔者将从Stacking 的工程原理上解释Stacking具体是怎么样的流程。

读懂下面的图，Stacking的原理也就了解啦

![img](https://i.stack.imgur.com/E5WGW.png)

### 明确符号

首先，来明确一下图中符号的概念

1、熟悉机器学习的都知道，我们有留一法和交叉验证。一般来说，我们先选用留一法，划分出Train 和Test。下面所有的模型都是基于Train得出的，并且将在Test上测试效果

2、如下图，Classfication models表示的是：已经训练好的第一层的模型，你想要进行整合的模型们。这里选用的是分类模型，所以是classfication model；

![image-20220828214816446](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220828214816446.png)

3、如下图，Predictions表示的是：第一层模型们基于你输进去的数据，产生的预测值

![image-20220828215020594](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220828215020594.png)



4、Meta-Classifier表示用作整合第一层模型的Final model。运用Meta-Classifier，你就可以得到最终的预测



### 工作流程



1、将Train输入给Classfication models，每个model都会得到一个预测结果，分别为P1，P2···同样的，我们也可以在Test上做一样的事情，得到在Test上的预测结果，设为L1，L2···

2、将P1，P2···和Trian中的目标变量合并成一个新的数据集，如下图。矩阵Z由P1，P2等组成，y表示Train中原本目标变量的值（也就是实际值）。这样我们就得到了一个新的数据集，这个数据集中有：Classfication models们的预测结果（图中的Z），也有目标变量的实际值（图中的y）。我们设这个数据集为D

​	![image-20220828214213628](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220828214213628.png)

数据集D的形式：这里假设每个Classfication model的预测结果都是以“概率形式”输出的

![image-20220829150808106](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220829150808106.png)



同样的，把在Test上的预测结果L1，L2等等也合并成一个新的数据集，我们设为T。注意：这个T中是没有目标变量的实际值，也就是没有y。因为这是Test，拟合Test的目的就是查看模型performance

3、在刚刚产生的数据集D上，训练Meta-Classifier。这样，Meta-Classifier学习的是第一层模型们的预测结果和实际值的关系。至此，我们Stacking的模型训练过程就结束了。使用这个Meta—Classifier，你就可以得到Stacking后的结果了。

4、查看Meta-Classifier的performance：将刚刚在Test上的拟合结果——数据集T——输入给训练好的meta-Classifier，可以得到在Test上的最终预测。也就是说，你可以查看Meta-Classifier在Test上的performance了。



### 常见问题

1、**“一次性的输入Train和Test”和“在训练好Meta-Classifier后，再输入Test进行performance查看”有什么区别？**

答：在无cv的方法中，其实这是没有区别的。在训练好后，再输入Test进行查看performance，其实与上面讲的工作流程中关于Test的拟合，是一码事

2、**如何选择Meta-Classifier？**

答：一般来说，根据经验主义，我们会选择比较简单的分类模型，最常用的是Logistic regression（逻辑回归）。因为其实Meta-Classifier所要拟合的数据是比较简单的，复杂的模型更有可能会出现过拟合的情况，这是我们不想看到的

3、 **什么样的Stacking才是有效的？**

答：衡量一个Stacking是否有效很简单，Stacking 后的模型的performance不能比原先模型们的performance差。这里的performance指的是你自己选择的衡量模型好坏的指标。

4、**应该Stacking什么样的模型最好？**

答：第一层的模型之间差异性越大越好，这样Stacking的效果才越显著。例如，笔者在kaggle Titanic比赛中，选择Stacking了XGboost、svm、knn。因为XGboost是一个梯度提升树，是树形模型；SVM是基于线性划分的模型；knn是基于距离的模型。笔者认为三者在算法上是存在差异性的，也就是捕捉数据的能力和捕捉到的数据的特征是不一样的。最后得到Stacking结果，相较于单一的模型，确实有提高。

打个广告，欢迎大家查看笔者关于kaggle Titanic比赛写的kernel。如果觉得有帮助的话，不妨点个Upvotes。link：https://www.kaggle.com/code/xiaohu2200/reach-top-100-by-xgboost-and-stacking-0-8245



# Cross-Validation  Stacking

众所周知，Cross-Validation（交叉验证）是一种避免模型过拟合的方法。通过在Stacking中使用交叉验证，使得Meta-Classifier将要拟合的数据更加复杂，减小Meta-Classifier过拟合的可能性，提升其泛化能力。当然，想要避免过拟合，你可以在Meta-Classifier的目标函数中增加L1，L2正则化项。

同样的，我们从一张图进行解释

![image-20220829162000488](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220829162000488.png)

### 明确符号

1、图中的Training和Holdout，分别就是Train和Test

2、图中展示的是k-fold=5的情况，并且这是对于第一层模型的拟合



### 工作流程

1、先根据k-fold数量，将Train划分为5份，每一份我们都称之为该k折的Validation（验证集）。就是交叉验证的基本操作。Test不做变动

2、对于每个第一层的model，从k=1开始，都在Train划分的Training（上图中的绿色部分）上训练模型，在该k折的Validation（验证集）上进行预测。重点来了，保存在Validation（验证集）上的预测结果，将其作为第二层模型的输入数据。

其实，有cv与无cv的区别很简单。无cv是将一次性在Trian上得到预测结果，作为第二层的输入数据。而有cv是利用交叉验证，每次都基于该模型的超参数，在Training上重新拟合一个模型（instance），之后将在Validation上的预测结果，作为为第二层的输入数据。所以也就是说，第二层模型的输入数据，是基于5个不一样的模型（instance）得到的。当然，这些模型（instance）是基于同样的超参数，只是拟合的数据不一样。

举个例子：下图就是第二层模型的输入数据。我们关注P1这一列：k=1时候，P1的值是0.6。表明在k=1时，基于第一层的model1的超参数首先拟合了划分的Training（上图的绿色部分），得到一个模型（instance），然后该模型（instance）在Validation（验证集）上，做出的预测结果为0.6。同理，k=2··5都是这样的。所以，P1的这五行，是五个不同的模型（instance），在五个不同的Validation（验证集）上得到的结果。



![image-20220829163015170](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220829163015170.png)

那图中的Holdout呢？

Holdout其实就是Test，刚刚我们得到了第二层模型的输入数据，现在我们要得到第二层模型的Test数据。

刚刚提到，从k=1到5，我们得到了5个模型（instance）。我们将这个5个模型（instance）分别在Test上进行拟合，得到的数据应该是如下图的。下图表示的是，第一层模型中的model1，在Train中进行Cross-Validation时，得到的五个模型（instance），分别在Test上拟合的预测结果。这里的Test笔者假设只有3行。我们将5个模型（instance）的结果进行平均，得到的Final，就可以作为第二层模型的Test（Holdout）了。

![image-20220829165637490](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220829165637490.png)

将所有第一层model的Final整合起来，就得到第二层的Test了，如下图

![image-20220829165800766](https://typora-nigel.oss-cn-nanjing.aliyuncs.com/img/image-20220829165800766.png)

3、到这里，我们得到了：a. 第二层模型的Train数据；b.第二层模型的Test数据。

这里有两个做法：a.简单的：直接在Train上拟合，Test上检查performance；b.复杂的：也在Train上进行Cross-Validation，再在Test上检查performance

a.简单的：与无cv时第二层模型的训练方法一样，拟合第一层得到的Train，在Test上查看performance

b.复杂的：在拟合Train时，也使用Cross-Validation。只是，这时Cross-Validation进行的划分，需要按照第一层的划分来。比如，原本在第一层是在k=1的样本，在第二层的Cross-Validation也应该被划分在k=1中。这样，我们能得到第二层model在Train上，具有统计意义的performance。我们常常利用这些信息来看，该model在Train上的平均performance，这样的performance更具有普遍性。而在Test上，与简单的方法是一样的——直接进行拟合查看performance

4、使用得到的Meta-Classifier，就可以做出基于Stacking model 的预测



### 常见问题

1、**“一次性的输入Train和Test”和“在训练好Meta-Classifier后，再输入Test进行performance查看”有什么区别？**

答：与无cv不同，在有cv时是非常不一样的。如果我们**不是一次性的输入Test**：我们使用Stacking model在Test上进行拟合，是一个没有cv的过程。Test首先在第一层模型进行拟合，第一层模型的输出结果作为第二层模型的input，最后得到预测结果，也就是说所有的结果都是基于一个模型在Test上的拟合。而**一次性的输入Train和Test**，第一层模型的输出结果是基于每个model的5个模型（instance）得到的。一个是基于一个模型，一个是基于5个模型（k-fold=5时），这是非常不一样的。至于哪个好一点，笔者也说不清楚。但笔者发现，在python中的大多数Stacking的函数，都是只需要输入Trian 的，Train和Test是分开输入的

2、**在新的数据集上预测的时候，输入的数据会再次进行Cross-Validation吗？**

答：不会。Cross-Validation仅仅只是训练模型时所用到的方法。而在新的数据集上进行预测，是基于已经训练好的模型（instance）进行的。



## 写在最后

笔者也是一位在Data Science的海洋中不断前进的学生，水平不算太高，所以如果文章什么地方有纰漏或错误，欢迎大家指正！如果觉得笔者做到努力是有成效的话，不妨点个赞吧！



## 有关文献

Saso Džeroski,Bernard Ženko(2004). Is Combining Classifiers with Stacking Better than Selecting the Best One. *Machine Learning*, 54, 255–273, 2004.

(https://link.springer.com/content/pdf/10.1023/B:MACH.0000015881.36452.6e.pdf)

Funda Güneş, Russ Wolfinger, Pei-Yi Tan(2017). Stacked Ensemble Models for Improved Prediction Accuracy. *Paper SAS*,2017.

(http://support.sas.com/resources/papers/proceedings17/SAS0437-2017.pdf)

带有代码的博客

1、https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html；

2、https://mlfromscratch.com/model-stacking-explained/

3、https://bradleyboehmke.github.io/HOML/stacking.html

