import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from dataclasses import dataclass
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Initiating model trainer")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )

            models = {
                LogisticRegression : LogisticRegression(),
                KNeighborsClassifier : KNeighborsClassifier(),
                DecisionTreeClassifier : DecisionTreeClassifier(),
                SVC : SVC(),
                RandomForestClassifier : RandomForestClassifier(),
                AdaBoostClassifier : AdaBoostClassifier(),
                GradientBoostingClassifier : GradientBoostingClassifier(),
                XGBClassifier : XGBClassifier(),
                CatBoostClassifier : CatBoostClassifier(),
            }

            params = {
                LogisticRegression: {
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg'],
                    'max_iter': [100, 200, 500],
                    'fit_intercept': [True, False],
                    'class_weight': [None, 'balanced']
                },
                KNeighborsClassifier: {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [10, 20, 30, 40],
                    'p': [1, 2]
                },
                DecisionTreeClassifier: {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [None, 3, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': [None, 'sqrt', 'log2']
                },
                SVC: {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'degree': [2, 3, 4],
                    'class_weight': [None, 'balanced']
                },
                RandomForestClassifier: {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'bootstrap': [True, False],
                    'class_weight': [None, 'balanced']
                },
                AdaBoostClassifier: {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.001, 0.01, 0.1, 1, 2]
                },
                GradientBoostingClassifier: {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10]
                },
                XGBClassifier: {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                },
                CatBoostClassifier: {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                    'depth': [3, 5, 7, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128],
                    'verbose': [0]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models = models,params = params)

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model
                )

            logging.info(f"best model name is {best_model_name} and its score is {best_model_score}")

            best_model.fit(X_train,y_train)
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test,y_pred)

            print(best_model)
            print("Accuracy score {:.4f}".format(acc))

            return acc
        
        except Exception as e:
            raise CustomException(e,sys)