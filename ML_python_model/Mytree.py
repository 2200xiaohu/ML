import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn import tree
from Smile.model import Mymodel

class Mytree(Mymodel.Mymodel):
    '''
    实现基于sklearn中决策树的封装

    功能
    1、创建树(默认使用cv)
    2、网格搜索
    '''
    def __init__(self, usewhat=0) -> None:
        '''
        继承基类Mymodel
        '''
        super(Mytree, self).__init__()
        self.model = tree.DecisionTreeClassifier()
        self.grid_model = tree.DecisionTreeClassifier()
    
    # def fit(self, x, y):
    #     if (self.fitwhat == 0):
    #         Mytree.cv_fit(x, y)
    #     elif (self.fitwhat == 1):
    #         Mytree.grid_search(x, y, self.parameters)
    #     return self

    def cv_fit(self, x=0, y=0, k=5):
        '''
        实现交叉验证
        最后model实现fit
        '''
        super().cv_fit(x, y, k)
        # kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        # scores = cross_val_score(self.model, x, y, cv=kf)
        # print(scores)
        # self.model.fit(x, y)

    def grid_search(self, x, y, parameters={}, k=5):
        '''
        parameters:格式为字典，values为列表形式存储超参数
        最后设置超参数结果模型
        '''
        super().grid_search(x, y, parameters, k)
        # cross_valid_scores = {}
        # start = time.time()
        # model_gridSearch_tree = GridSearchCV(self.model, parameters, cv=self.k, scoring='accuracy', verbose=3, return_train_score=True, n_jobs=-1)
        # model_gridSearch_tree.fit(x, y)
        # end = time.time()
        # print('-----')
        # print("Time=", end-start)
        # print(f'Best parameters {model_gridSearch_tree.best_params_}')
        # print(
        #     f'Mean cross-validated accuracy score of the best_estimator: ' + 
        #     f'{model_gridSearch_tree.best_score_:.3f}'
        # )
        # cross_valid_scores['tree'] = model_gridSearch_tree.best_score_
        # print('-----')
        # #设置超参数结果模型
        # self.grid_model.set_params(**model_gridSearch_tree.best_params_)
        # self.fitwhat = 1
        # self.parameters = parameters

