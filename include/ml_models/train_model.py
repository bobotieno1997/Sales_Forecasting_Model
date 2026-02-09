import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import yaml
import joblib
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import optuna
import mlflow

from utils.mlflow_utils import MLflowManager
from feature_engineering.feature_pipeline import FeatureEngineer
from data_validation.validators import DataValidator

# from ml_models.advanced_ensemble import AdvancedEnsemble
# from ml_models.diagnostics import diagnose_model_performance
# from ml_models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path : str = '/usr/local/airflow/include/config/ml_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.modle_config = self.config('models')
        self.training_config = self.config('training')
        self.mlflow_manaer = MLflowManager(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.data_validator = DataValidator(config_path)


        self.models ={}
        self.scaler = {}
        self.encoders = {}

    def prepare_data(self,df: pd.DataFrame, target_col:str = 'sales',
                     date_col: str = 'date', group_cols: Optional[List[str]] = 'None',
                     categorical_cols: Optional[List[str]] = None):
        
        logger.info('Preparing data for training')

        required_cols = ['date', target_col]
        if group_cols: 
            required_cols.extend(group_cols)

        missing_cols = set( required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        

        df_features = self.features_engineering.create_all_features(
            df, target_col=target_col,date_col=date_col,
            group_cols=group_cols, categorical_cols=categorical_cols
        )


        # Split data chronologically for the timeseries
        df_sorted = df_features.sort_values(by=date_col)


        train_size = int(len(df_sorted) * (1 - self.training_config['test_size'] - self.training_config['validation_size']))
        val_size = int(len(df_sorted) * self.training_config['validation_size'] )


        train_df = df_sorted[:train_size]
        val_df = df_sorted[train_size:train_size+val_size]
        test_df = df_sorted[train_size+val_size:]


        train_df = train_df.dropna(subset=['target_col'])
        val_df = val_df.dropna(subset=['target_col'])
        test_df = test_df.dropna(subset=['target_col'])


        logger.info(f"Train data split: {len(train_df)}, Validation data split: {len(val_df)}, Test data split: {len(test_df)}")

        return train_df, val_df, test_df
    

    def preprocess_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                            test_df: pd.DataFrame, target_col: str,
                            exclude_col: List[str] = ['date']):
        
        logger.info('Preprocessing features')
        feature_cols = [col for col in train_df.colums if not exclude_col + [target_col]]


        x_train = train_df[feature_cols].copy()
        x_val = val_df[feature_cols].copy()
        x_test = test_df[feature_cols].copy()


        y_train = train_df[target_col].values()
        y_val = val_df[target_col].values()
        y_test = test_df[target_col].values()


        # Encoding
        categorical_cols = x_train.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col]=LabelEncoder()
                x_train.loc[:,col]= self.encoder[col].fit_transform(x_train[col].astype(str))
            else:
                x_train.loc[:,col] = self.encoders[col].transform(x_val[col].astype(str))

        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(x_train)
        X_val_scaled = scaler.fit_transform(x_val)
        X_test_scaled = scaler.fit_transform(x_test)

        # Convert back to DataFrame to preserve feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=x_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=x_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=x_test.index)

        self.scalers['standard'] = scaler
        self.feature_cols = feature_cols
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


