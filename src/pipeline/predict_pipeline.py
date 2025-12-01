import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

# ---------- Paths & Caching ----------

MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

_model = None
_preprocessor = None


def get_model():
    """Load model once and cache it."""
    global _model
    if _model is None:
        try:
            logging.info(f"Loading model from {MODEL_PATH}")
            _model = load_object(MODEL_PATH)
        except Exception as e:
            raise CustomException(e, sys)
    return _model


def get_preprocessor():
    """Load preprocessor once and cache it."""
    global _preprocessor
    if _preprocessor is None:
        try:
            logging.info(f"Loading preprocessor from {PREPROCESSOR_PATH}")
            _preprocessor = load_object(PREPROCESSOR_PATH)
        except Exception as e:
            raise CustomException(e, sys)
    return _preprocessor


# ---------- Prediction Pipeline ----------

class PredictPipeline:
    def __init__(self):
        # Heavy objects are loaded lazily via helpers above
        pass

    def predict(self, features: pd.DataFrame):
        try:
            preprocessor = get_preprocessor()
            model = get_model()

            # Transform input
            data_scaled = preprocessor.transform(features)

            # Model prediction
            pred = model.predict(data_scaled)

            # Lightweight "explanation" using feature importances if available
            feature_names = features.columns.tolist()

            importances = None
            try:
                # For tree-based models (CatBoost, RandomForest, etc.)
                importances = getattr(model, "feature_importances_", None)
            except Exception:
                importances = None

            if importances is not None and len(importances) == len(feature_names):
                explanation = sorted(
                    zip(feature_names, importances),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
            else:
                # Fallback: zero importance, just to keep template logic simple
                explanation = [(name, 0.0) for name in feature_names]

            return pred, explanation

        except Exception as e:
            raise CustomException(e, sys)


# ---------- Input Data Wrapper ----------

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
