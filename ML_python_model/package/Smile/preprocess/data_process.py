import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
# Import data
class Mytransform():
    def __init__(self):
        self.transform_dict = {}
        self.train_or_test = 0

    def load_data(self, path, type):
        if(type=='csv'):
            return pd.read_csv(path)
        elif(type=='excel'):
            return pd.read_excel(path)

    # def fit(self, x, y=None):
    #     if(self.train_or_test == 0):
    #         data = Mytransform.transform(self, x)
    #     else:
    #         data = Mytransform.fit_transform(self, x)
    #     return data

    #转换数据
    def transform(self, data):
        #保持转换的参数
        col = data.columns
        #标准化
        normaler = Normalizer()
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = normaler.fit_transform(data)

        self.transform_dict[scaler] = scaler
        self.transform_dict[normaler] = normaler

        data = pd.DataFrame(data, columns=col)
        return data

    def fit_transform(self, data):
        col = data.columns
        for t in self.transform_dict.values():
            data = t.transform(data)
        data = pd.DataFrame(data, columns=col)
        return data