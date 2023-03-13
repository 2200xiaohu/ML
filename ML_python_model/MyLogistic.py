import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from Smile.model import Mymodel

class MyLogistic(Mymodel.Mymodel):
    def __init__(self, usewhat=0) -> None:
        '''
        usewhat 使用什么
        cv_fit -- 0
        grid_search -- 1
        '''
        self.model = LogisticRegression()
        self.grid_model = LogisticRegression()
    
    def cv_fit(self, x=0, y=0, k=5):
        '''
        实现交叉验证
        最后model实现fit
        '''
        super().cv_fit(x, y, k)
    
    def grid_search(self, x, y, parameters={}, k=5):
        '''
        parameters:格式为字典，values为列表形式存储超参数
        最后设置超参数结果模型
        '''
        super().grid_search(x, y, parameters, k)

