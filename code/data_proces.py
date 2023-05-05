import utils
import pandas as pd
from config import ConfigPath
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import defaultdict
import config

class DataTransform:
    @staticmethod
    def split_xy(data:pd.DataFrame, target, columns=ConfigPath().list_of_columns):
        y = None
        if target in data.columns:
            y = data[target]
        X = data[columns]
        return X, y
    
    
    def __init__(self, 
                 path_LabelEncoder=config.ConfigPath().path_LabelEncoder,
                 path_Scaler=config.ConfigPath().path_Scaler,
                 target=config.ConfigPath().target):
        
        self.path_LabelEncoder = path_LabelEncoder
        self.path_Scaler = path_Scaler
        
        self._columns_categorial = None
        self.target = target
        
    def fit_transform(self, data:pd.DataFrame):
        
        if config.ConfigPath().clean_data:
            data = utils.clean_data(data)
        
        d = defaultdict(LabelEncoder)
        scaler = MinMaxScaler()
        columns_categorial  = [name for i,name in enumerate(data.columns) if data.dtypes[i] == object]
        self._columns_categorial = columns_categorial
        if columns_categorial:
            data[columns_categorial] = data[columns_categorial].apply(lambda x: d[x.name].fit_transform(x))
        
        X, y = DataTransform.split_xy(data, self.target)
        X = scaler.fit_transform(X)
        utils.save_pikle(scaler, self.path_Scaler)
        utils.save_pikle(d, self.path_LabelEncoder)
        return X, y
        
    def transform(self, data:pd.DataFrame):
    
        if config.ConfigPath().clean_data:
            data = utils.clean_data(data)
    
        X, y = DataTransform.split_xy(data, self.target)
        d = utils.load_pikle(self.path_LabelEncoder)    
        scaler = utils.load_pikle(self.path_Scaler)    
        columns_categorial  = [name for i,name in enumerate(data.columns) if data.dtypes[i] == object]
        if columns_categorial:
            data[columns_categorial] = data[columns_categorial].apply(lambda x: d[x.name].transform(x))    
        X = scaler.transform(X)
        return X, y