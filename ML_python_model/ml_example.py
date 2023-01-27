# -*- coding: utf-8 -*-
"""ML_example.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RAmyHxxDtMmlY41jTdlF_7pH40OvkBMv

#划分训练集和数据预处理

##数据预处理

###标准化和归一化
"""

data=pd.read_csv('C:/Users/nigel/Desktop/kaggle competition/playground1/data/data_omit.csv')#导入数据
cols = data.columns[data.columns != 'failure']
y=pd.DataFrame(data['failure'])#y为目标变量
x=data[cols]
#转化格式
data['failure']=data['failure'].astype('category',errors='ignore')#转换类型

#标准化
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
normaler = Normalizer()
scaler = StandardScaler()
cols =x.columns
x[cols] = scaler.fit_transform(x[cols])
x[cols] = normaler.fit_transform(x[cols])

"""## 划分训练集"""

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    

SEED = 42
set_seed(SEED)
#划分训练集
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=,trian_size=,random_state,shuffle,stratify)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=SEED)

"""# 训练模型

## 创建学习器 交叉验证选择超参数

不同模型只需要更换学习器和参数
"""

1、使用sklearn中的学习器
2、使用GridSearchCV进行交叉验证

import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
cross_valid_scores = {}
start=time.time()
parameters = {
    "C": np.arange(0.1,0.2,step=0.01),
    "kernel": ["linear","sigmoid"],# "poly", "rbf", sigmoid
    "gamma": ["scale", "auto"],
}

model_svc = SVC(
    random_state=SEED,
    class_weight="balanced",
    probability=True,
)

model_svc = GridSearchCV(
    model_svc, 
    parameters, 
    cv=10,
    scoring='accuracy',
    verbose=3
)


svc_model=model_svc.fit(x_train, y_train.values.ravel())
end=time.time()
print('-----')
print("Time=",end-start)
print(f'Best parameters {model_svc.best_params_}')
print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + 
    f'{model_svc.best_score_:.3f}'
)
cross_valid_scores['svc'] = model_svc.best_score_
print('-----')

import xgboost as xgb
import time
start=time.time()
parameters = {
    'max_depth': [3, 5, 7, 9], 
    'n_estimators': [5, 10, 15, 20, 25, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'gamma':[0],
    'max_delta_step':[0],
    'min_child_weight':[1],
    'min_split_loss':[0.01]
    
}

model_xgb = xgb.XGBClassifier(
    random_state=SEED,
    objective='binary:logistic',
    tree_method='gpu_hist',
    predictor='gpu_predictor',n_jobs=2
)

model_xgb = GridSearchCV(
    model_xgb, 
    parameters, 
    cv=10,
    scoring='accuracy',
    verbose=3
)

xgb_model=model_xgb.fit(x_train, y_train.values.ravel())
end=time.time()
print('-----')
print("Time=",end-start)
print(f'Best parameters {model_xgb.best_params_}')
print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + 
    f'{model_xgb.best_score_:.3f}'
)
cross_valid_scores['xgboost'] = model_xgb.best_score_
print('-----')

"""## 设置超参数"""

best_param=svc_model.best_params_#获取最优超参数
svc_model_hyper=SVC(
    random_state=SEED,
    class_weight="balanced",
    probability=True,
)#创建新学习器
svc_model_hyper.set_params(**best_param)#传递参数

best_param_xgboost=xgb_model.best_params_
parameters_stack_xgboost = {
    'max_depth': [9], 
    'n_estimators': [25],
    'learning_rate':  [0.1],
    'gamma':[0.1],
    'max_delta_step':[0],
    'min_child_weight':[1],
    'min_split_loss':[0.01]
    
}
xgb_model_hyper = xgb.XGBClassifier(
    random_state=SEED,
    objective='binary:logistic',n_jobs=2
)

xgb_model_stack.set_params(**best_param_xgboost)

"""## 通过kfold获得各个模型在相同数据上的效果，比较选择模型"""

#保存将要比较的模型
#传递的是设置好超参数的模型
models={
    'svm':svc_model_hyper,
    'Xgboost':xgb_model_hyper
}

from sklearn.model_selection import KFold, cross_val_score
#设置函数
def kf_cross_val(model,X,y):
    
    scores,feature_imp, features = [],[], []
    
    kf = KFold(n_splits=5,shuffle = True, random_state=42)
    
    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        
        x_train = X.loc[train_index]
        y_train = y.loc[train_index]
        x_test = X.loc[test_index]
        y_test = y.loc[test_index]
        
        model.fit(x_train,y_train)
        
        y_pred = model.predict_proba(x_test)[:,1]     # edit 
        scores.append(roc_auc_score(y_test,y_pred))
        
        try:
            feature_imp.append(model.feature_importances_)
            features.append(model.feature_names_)
        except AttributeError: # if model does not have .feature_importances_ attribute
            pass
        
    return feature_imp, scores, features
 
results  = {}
 
for name,model in models.items():
    
    feature_imp,result,features = kf_cross_val(model, x_train, y_train)
    results[name] = result
 
for name, result in results.items():
    print("----------\n" + name)
    print(np.mean(result))
    print(np.std(result))
    print(feature_imp)

"""## Stacking

将之前设置好的学习器传入
有两种stack： 1、每个学习器都设置好了超参数，则直接进行训练；2、在stacking中一起进行超参数选择
"""

import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlxtend.regressor import StackingCVRegressor
from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection
from cuml.linear_model import LogisticRegression
lr = LogisticRegression()
stack = StackingCVClassifier(classifiers=[xgb_model_hyper,svc_model_hyper],
                            meta_classifier=lr, cv=10,
                            use_features_in_secondary=False,
                            store_train_meta_features=True,
                            shuffle=False,n_jobs=2)

# Commented out IPython magic to ensure Python compatibility.
for clf, label in zip([xgb_model_hyper, svc_model_hyper,stack], 
                      ['xgb', 
                       'svc', 
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf,x_train , y_train,cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
#           % (scores.mean(), scores.std(), label))

result=stack.fit(x_train.values ,y_train.values)

"""# 检测效果"""

from sklearn import metrics
pred = stack.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test, pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

## Import the library and functions you need
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
## Accuracy
print("Accuracy:",accuracy_score(y_test,pred))
## Precision
print("Precision:",precision_score(y_test,pred))
## Recall
print("Recall:",recall_score(y_test,pred))
## F1 Score
print("F1:",f1_score(y_test,pred))