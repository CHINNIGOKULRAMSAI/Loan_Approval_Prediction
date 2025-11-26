import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_path : str = os.path.join("artifacts","train.csv")
    test_path : str = os.path.join("artifacts","test.csv")
    raw_path : str = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("now entered into data ingestion")
        try:
            df = pd.read_csv("notebook/data/cleaned_loan_approval_dataset.csv")
            logging.info("Reads Traning and test data as dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_path),exist_ok=True)
            
            df.to_csv(self.data_ingestion_config.raw_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.25,random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_path,index=False,header=True)

            logging.info("ingeston of data is completed")
            return (
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_path,test_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)
            