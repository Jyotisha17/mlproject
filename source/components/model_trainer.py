import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from source.exception import customException
from source.logger import logging

from source.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spiliting training and test data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "RandomForest Regressor":RandomForestRegressor(),
                "DecisionTree Regressor":DecisionTreeRegressor(),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "k-Neighbors Regressor":KNeighborsRegressor(),
                "XGB Regressor":XGBRegressor(),
                "CatBoost Regressor ":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor ":AdaBoostRegressor()
            } 
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,
                                             y_test=y_test,models=models)
            
            ## To get best model score from dictonary
            best_model_score = max(sorted(model_report.values()))
            
            ## To get best model name from dict
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise customException("No best model found")
            logging.info(f"Best found model on training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(x_test)
            
            r2score=r2_score(y_test,predicted)
            return r2score
        
        except Exception as e:
            raise customException(e,sys)
        
    
