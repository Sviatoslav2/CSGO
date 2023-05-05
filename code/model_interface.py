from config import ConfigPath
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_proces import DataTransform
import pandas as pd
import os
from sklearn.metrics import classification_report
from utils import load_pikle, save_pikle


class Model_interface:
    def __init__(self):
        
        self.path_model = ConfigPath().path_model
        if os.path.exists(self.path_model):
            self.model = load_pikle(self.path_model)
        else:
            self.model = None
    
    def fit(self, X, y): #### 
        self.model = RandomForestClassifier(n_estimators=20)
        self.model.fit(X, y)
        save_pikle(self.model, self.path_model)
        
    
    def fit_dataframe(self, data:pd.DataFrame):
        data_transform = DataTransform()
        X, y = data_transform.fit_transform(data)
        self.model = RandomForestClassifier(n_estimators=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        save_pikle(self.model, self.path_model)
    
    def __call__(self, X):
        if not os.path.exists(self.path_model):
            raise ErrorFit("fit shuld be used first")
        self.model = load_pikle(self.path_model)
        return self.model.predict(X)
        
    def predict_prob(self, X):
        if not os.path.exists(self.path_model):
            raise ErrorFit("fit shuld be used first")
        self.model = load_pikle(self.path_model)
        return self.model.predict_proba(X)
        
    def predict_df(self, data:pd.DataFrame):
        self.model = load_pikle(self.path_model)
        data_transform = DataTransform()
        X, y = data_transform.transform(data)
        predict_prob_value = self.predict_prob(X)
        
        predict_value = self.model.predict(X)
        data["predict_prob_value0"] = predict_prob_value[:,0]
        data["predict_prob_value1"] = predict_prob_value[:,1]
        data["predict_value"] = predict_value
        return data
        