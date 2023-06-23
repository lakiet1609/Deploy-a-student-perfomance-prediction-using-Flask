import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifact', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_transformed(self):
        try:
            numerical_features = ['writing score', 'reading score']
            nominal_features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course']
            ordinal_features = ['parental level of education']
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            education_values = ["master's degree", "bachelor's degree", "associate's degree", 'some college', 'high school', 'some high school']
            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('ord_encoder', OrdinalEncoder(categories=[education_values]))
            ])
            
            nominal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('nom_encoder', OneHotEncoder(sparse_output=False))
            ])
            
            logging.info('Numerical columns transformation completed')
            logging.info('Nominal columns transformation completed')
            logging.info('Ordinal columns transformation completed')
            
            preprocessor = ColumnTransformer(transformers=[
                ('num_features', num_pipeline, numerical_features),
                ('nom_features', nominal_pipeline, nominal_features),
                ('ord_features', ordinal_pipeline, ordinal_features), 
            ])
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            
            logging.info('Obtaining preprocessing data')
            
            preprocessing_data = self.get_transformed()
            
            target_features = 'math score'
            
            X_train = train_df.drop(columns=[target_features], axis=1)
            y_train = train_df[target_features]
            
            X_test = test_df.drop(columns=[target_features], axis=1)
            y_test = test_df[target_features]
            
            logging.info('Apply preprocessing on train and test dataframe')
            
            X_train_transform = preprocessing_data.fit_transform(X_train)
            X_test_transform = preprocessing_data.transform(X_test)
            
            train_array = np.c_[X_train_transform, np.array(y_train)]
            test_array = np.c_[X_test_transform, np.array(y_test)]
            
            logging.info('Saved preprocessing object')
            
            save_obj(file_path = self.data_transformation_config.preprocessor_obj_path, 
                     obj = preprocessing_data)
            
            return (train_array, test_array, self.data_transformation_config.preprocessor_obj_path)
            
        except Exception as e:
            raise CustomException(e,sys)