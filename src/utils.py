import os
import sys
import pickle
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

def save_object(file_path,obj):
    try:

        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,X_test,y_train,y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = RandomizedSearchCV(
                model,
                param_distributions=param,
                n_iter=20,
                scoring='accuracy',
                cv=5,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )

            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy_score = accuracy_score(y_train,y_train_pred)
            test_accuracy_score = accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_accuracy_score

        return report
            
    except Exception as e:
        raise CustomException(e,sys)