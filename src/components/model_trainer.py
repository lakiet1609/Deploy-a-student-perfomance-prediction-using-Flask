import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models
from sklearn.svm import LinearSVR
from sklearn.linear_model import RidgeCV, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    traned_model_path = os.path.join('artifact', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split the dataset into train set, test set')
            X_train, y_train, X_test, y_test = (train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1])
            
            models = {
                'Linear SVR': LinearSVR(max_iter=100000),
                'Ridge CV': RidgeCV(),
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(max_iter=100000),
            }
            
            params = {
                "Linear SVR":{
                    'epsilon': [0.1, 0.01, 0.001],
                    'C': [0.1, 0.5, 1, 5, 10],
                    'tol': [0.1, 0.01, 0.001],
                    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
                },
                
                "Ridge CV": {
                    'alphas': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                    'gcv_mode': ['auto', 'svd', 'eigen']
                },
                
                "Linear Regression": {},
                
                "Ridge": {
                    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                    'tol': [0.1, 0.01, 0.001],
                    'solver': ['auto', 'lsqr', 'sparse_cg', 'sag', 'saga']
                }           
            }
     

            
            train_report, test_report = evaluate_models(X_train=X_train,
                                                        X_test=X_test,
                                                        y_train=y_train,
                                                        y_test=y_test,
                                                        models=models,
                                                        params=params)
            
            rank_train_score = sorted(train_report.items(), key=lambda x:x[1][0], reverse=True)
            
            rank_test_score = sorted(test_report.items(), key=lambda x:x[1][0], reverse=True)
            
            best_model_name = rank_test_score[0][0]
            best_model = models[best_model_name]
            
            logging.info('Found the best model')
            
            save_obj(
                file_path=self.model_trainer_config.traned_model_path,
                obj=best_model
            )
            
            return rank_train_score, rank_test_score, best_model_name
                    
        except Exception as e:
            raise CustomException(e,sys)


