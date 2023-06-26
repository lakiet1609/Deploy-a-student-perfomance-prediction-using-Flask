import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv(r'notebook\data\exams.csv')
            
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train test split initiated')
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info('Ingestion of the data is completed')
        
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initate_data_ingestion()
    
    data_transform = DataTransformation()
    train_array, test_array, _ = data_transform.initiate_data_transformation(train_path=train_data,
                                                                             test_path=test_data)
    
    model_trainer = ModelTrainer()
    
    train_rank, test_rank, best_model = model_trainer.initiate_model_trainer(train_array=train_array, 
                                                                             test_array=test_array)
    
    print(f'Evaluation of the best model: {best_model} on training set: {train_rank[0]}, on testing set: {test_rank[0]}')
    

    
    
    