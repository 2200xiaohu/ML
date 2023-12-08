## 神经网络的CV

早停可能会出现有该epoch过拟合的情况，就是当前epoch的score大于周围的epoch

所有更好的是每次训练都固定轮数，最后的一个epoch作为模型



**如何确定这个轮数呢？**

1、对模型做5折（10折也行），每一折都用上早停，记录早停的轮数

2、取所有折早停轮数的均值，作为固定轮数



## 何时使用ranker模型

1. 检索的时候，先粗排再用ranker模型精排。 例子：Kaggel LLM
2. 模型得到许多点的预测，需要筛选缩小范围的时候



## 输入特征有numerical和category变量

可以分别通过不同的embedding层，再concat到一起[方案](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459715)





# Kaggle Sleep

#### What to learn

- [2nd stage: limit the number of events per day](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459627)

- [Weighted Box Fusion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459637)
- [wavenet](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459598), [Unet](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459637)

- [BiLSTM](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459604)



# Kaggle LLM

- [ ] 通过检索扩充问题，关键点在于

  - [ ] 如何做embedding
    - [ ] tf-idf / BM25：用cuml做更快
    - [ ] 使用embedding model
    - [ ] 精度：最好是将一段话拆成多个句子，将粒度变小
  - [ ] 如何检索：
    - [ ] faiss包的根据embedding进行检索
    - [ ] tfidf / BM25 的计算
    - [ ] 使用Ranker模型
      - [ ] 一个要点：必须知道question对应的实际context是什么
      - [ ] XGBRanker，LGBRanker，debertav3
      - [ ] 数据集的构建：
        1. 根据gpt生成的问题，可以将传入gpt的文章作为Truth
        2. 训练数据类似于：(question, passage) —— >  label: 0 / 1
        3. 也可以通过对passage做特征，例如n-gram，输入给LGBRanker
    - [ ] [训练代码](https://www.kaggle.com/code/podpall/3rd-place-reranker-training)
  - [ ] 总结：可以先通过embedding做一次筛选，然后再用BM25做二次筛选
- [ ] 模型使用debertaV3，加入AMP
- [ ] embedding - model使用 simcse重新训练提高效果，加入困难样本对比**Difficult Sample Comparison Learning**
- [ ] 检索用的数据集很重要
- [ ] 拆分文章为句子，利用包spacy
- [ ] 模型集成
  - [ ] [使用XGBRanker模型集成](https://www.kaggle.com/code/sorokin/llm-xgboost-abc#%F0%9F%94%A8-XGBoost)
  - [ ] 输入的特征为：多个模型的对每个answer的预测结果，以及多个模型的分数
  - [ ] 等于是输入：问题一，（答案A，特征）—— > label : 问题一的真实答案
  - [ ] 结果：每个问题的答案排序

