import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts","preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def preprocessor_func(self):
        try:
            num_features = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
            cat_features = ['education', 'self_employed']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("std_scaler",StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("oneHotencoder",OneHotEncoder()),
                    ("std_scaler",StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Preprocessing is initiated")

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,num_features),
                ("cat_pipeline",cat_pipeline,cat_features),
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data is completed")
            
            preprocessing_obj = self.preprocessor_func()

            target_column = ['loan_status']

            input_fea_train_df = train_df.drop(columns=target_column,axis=1)
            target_fea_train_df = train_df[target_column]

            input_fea_test_df = test_df.drop(columns=target_column,axis=1)
            target_fea_test_df = test_df[target_column]

            logging.info("Applying preprocessing on training and testing dataframe")

            train_df_arr = preprocessing_obj.fit_transform(input_fea_train_df)
            test_df_arr = preprocessing_obj.transform(input_fea_test_df)

            train_arr = np.c_[train_df_arr,np.array(target_fea_train_df)]
            test_arr = np.c_[test_df_arr,np.array(target_fea_test_df)]

            logging.info("Saved preprocessing process")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessing_obj,
                )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path,
                )
        
        except Exception as e:
            raise CustomException(e,sys)

