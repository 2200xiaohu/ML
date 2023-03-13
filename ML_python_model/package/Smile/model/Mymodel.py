import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
class Mymodel():
    def __init__(self, usewhat=0) -> None:
        '''
        所有自定义model的基类
        ————————————————————
        usewhat 使用什么
        cv_fit -- 0
        grid_search -- 1
        '''
        self.model = None
        self.seed = 42
        self.grid_model = None
        self.fitwhat = usewhat
        self.k = 5
        self.parameters = {'random_state': self.seed}
    
    def cv_fit(self, x=0, y=0, k=5, parameters={}):
        '''
        实现交叉验证
        最后model实现fit
        '''
        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        self.model.set_params(**parameters)
        print(self.model)
        scores = cross_val_score(self.model, x, y, cv=kf)
        print(scores)
        self.model.fit(x, y)
    
    def grid_search(self, x, y, parameters={}, k=5):
        '''
        parameters:格式为字典，values为列表形式存储超参数
        最后设置超参数结果模型
        '''
        print(self.grid_model)
        cross_valid_scores = {}
        start = time.time()
        model_gridSearch = GridSearchCV(self.model, parameters, cv=self.k, scoring='accuracy', verbose=3, return_train_score=True, n_jobs=-1)
        model_gridSearch.fit(x, y)
        end = time.time()
        print('-----')
        print("Time=", end-start)
        print(f'Best parameters {model_gridSearch.best_params_}')
        print(
            f'Mean cross-validated accuracy score of the best_estimator: ' + 
            f'{model_gridSearch.best_score_:.3f}'
        )
        cross_valid_scores['tree'] = model_gridSearch.best_score_
        print('-----')
        #设置超参数结果模型
        self.grid_model.set_params(**model_gridSearch.best_params_)
        self.fitwhat = 1
        self.parameters = parameters

#SVM
from sklearn.svm import SVC

class MySVM(Mymodel):
    def __init__(self, usewhat=0) -> None:
        '''
        usewhat 使用什么
        cv_fit -- 0
        grid_search -- 1
        '''
        super(MySVM, self).__init__()
        self.model = SVC()
        self.grid_model = SVC()
    
    def cv_fit(self, x=0, y=0, k=5,parameters={}):
        '''
        实现交叉验证
        最后model实现fit
        '''
        super().cv_fit(x, y, k,parameters)
    
    def grid_search(self, x, y, parameters={}, k=5):
        '''
        parameters:格式为字典，values为列表形式存储超参数
        最后设置超参数结果模型
        '''
        super().grid_search(x, y, parameters, k)

from sklearn import tree

class Mytree(Mymodel):
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

    def cv_fit(self, x=0, y=0, k=5, parameters={}):
        '''
        实现交叉验证
        最后model实现fit
        '''
        super().cv_fit(x, y, k,parameters)

    def grid_search(self, x, y, parameters={}, k=5):
        '''
        parameters:格式为字典，values为列表形式存储超参数
        最后设置超参数结果模型
        '''
        super().grid_search(x, y, parameters, k)

from sklearn.linear_model import LogisticRegression

class MyLogistic(Mymodel):
    def __init__(self, usewhat=0) -> None:
        '''
        usewhat 使用什么
        cv_fit -- 0
        grid_search -- 1
        '''
        super(MyLogistic, self).__init__()
        self.model = LogisticRegression()
        self.grid_model = LogisticRegression()
    
    def cv_fit(self, x=0, y=0, k=5, parameters={}):
        '''
        实现交叉验证
        最后model实现fit
        '''
        super().cv_fit(x, y, k,parameters)
    
    def grid_search(self, x, y, parameters={}, k=5):
        '''
        parameters:格式为字典，values为列表形式存储超参数
        最后设置超参数结果模型
        '''
        super().grid_search(x, y, parameters, k)


from sklearn.neighbors import KNeighborsClassifier as knn

class MyKnn(Mymodel):
    def __init__(self, usewhat=0) -> None:
        '''
        usewhat 使用什么
        cv_fit -- 0
        grid_search -- 1
        '''
        super(MyKnn, self).__init__()
        self.model = knn()
        self.grid_model = knn()
    
    def cv_fit(self, x=0, y=0, k=5, parameters={}):
        '''
        实现交叉验证
        最后model实现fit
        '''
        super().cv_fit(x, y, k,parameters)
    
    def grid_search(self, x, y, parameters={}, k=5):
        '''
        parameters:格式为字典，values为列表形式存储超参数
        最后设置超参数结果模型
        '''
        super().grid_search(x, y, parameters, k)