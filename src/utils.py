import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    try:
        train_report = {}
        test_report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model, param, cv=6, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            train_r2_score = r2_score(y_true=y_train, y_pred=y_train_pred)
            train_mse_score = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
            train_mae_score = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
            
            y_test_pred = model.predict(X_test)
            test_r2_score = r2_score(y_true=y_test, y_pred=y_test_pred)
            test_mse_score = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
            test_mae_score = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
            
            train_report[list(models.keys())[i]] = [train_r2_score,train_mse_score,train_mae_score]
            
            test_report[list(models.keys())[i]] = [test_r2_score,test_mse_score,test_mae_score]
        
        return train_report, test_report

            
    except Exception as e:
        raise CustomException(e,sys)
    
def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)