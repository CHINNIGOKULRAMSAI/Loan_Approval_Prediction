import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


# Input Data Wrapper

class CustomData:
    def __init__(
        self,
        loan_id,
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
    ):
        self.loan_id = loan_id
        self.no_of_dependents = int(no_of_dependents)
        self.education = education
        self.self_employed = self_employed
        self.income_annum = float(income_annum)
        self.loan_amount = float(loan_amount)
        self.loan_term = int(loan_term)
        self.cibil_score = int(cibil_score)
        self.residential_assets_value = float(residential_assets_value)
        self.commercial_assets_value = float(commercial_assets_value)
        self.luxury_assets_value = float(luxury_assets_value)
        self.bank_asset_value = float(bank_asset_value)

    def get_data_as_dataframe(self):
        data = {
            "loan_id": [self.loan_id],
            "no_of_dependents": [self.no_of_dependents],
            "education": [self.education],
            "self_employed": [self.self_employed],
            "income_annum": [self.income_annum],
            "loan_amount": [self.loan_amount],
            "loan_term": [self.loan_term],
            "cibil_score": [self.cibil_score],
            "residential_assets_value": [self.residential_assets_value],
            "commercial_assets_value": [self.commercial_assets_value],
            "luxury_assets_value": [self.luxury_assets_value],
            "bank_asset_value": [self.bank_asset_value],
        }
        return pd.DataFrame(data)
