import sys
path = ".\\code"
sys.path.insert(1, path)

import model_interface
import config
import os
import pandas as pd

def predict(df, model, scaler_path):
    model_inter = model_interface.Model_interface()
    return model_inter.predict_df(data)
    
