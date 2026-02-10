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

from ml_models.advanced_ensemble import AdvancedEnsemble
from ml_models.diagnostics import diagnose_model_performance
from ml_models.ensemble_model import EnsembleModel

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
    
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': r2_score(y_true, y_pred)
        }
        return metrics
    
    
    def train_xgboost(self, x_train: np.ndarray, y_train: np.ndarray,
                     x_val: np.ndarray, y_val: np.ndarray,
                     use_optuna: bool = True) -> xgb.XGBRegressor:
        
        logger.info("Training XGBoost model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42
                }
                
                params['early_stopping_rounds'] = 50
                model = xgb.XGBRegressor(**params)
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
                
                y_pred = model.predict(x_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))
            
            best_params = study.best_params
            best_params['random_state'] = 42
        else:
            best_params = self.model_config['xgboost']['params']
        
        best_params['early_stopping_rounds'] = 50
        model = xgb.XGBRegressor(**best_params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=True)
        
        self.models['xgboost'] = model
        return model



    def train_lightgbm(self, x_train: np.ndarray, y_train: np.ndarray,
                      x_val: np.ndarray, y_val: np.ndarray,
                      use_optuna: bool = True) -> lgb.LGBMRegressor:
        
        logger.info("Training LightGBM model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42,
                    'verbosity': -1,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt'
                }
                
                model = lgb.LGBMRegressor(**params)
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], 
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                
                y_pred = model.predict(x_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))
            
            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['verbosity'] = -1
        else:
            best_params = self.model_config['lightgbm']['params']
        
        model = lgb.LGBMRegressor(**best_params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], 
                 callbacks=[lgb.early_stopping(50)])
        
        self.models['lightgbm'] = model
        return model


    
    def train_all_models (self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          test_df: pd.DataFrame, target_col: str = 'sales',
                          use_optuna: bool = True):

        results = {}

        run_id = self.mlflow_manager.start_run(
            run_name=f"Sales_forcast_training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            tags={'model_type' : 'ensemble', 'use_optuna':str(use_optuna)}
        )

        try:
            x_train, x_val, x_test, y_train, y_val, y_test = self.preprocess_features(
            train_df, val_df, test_df, target_col=target_col   
            )

            self.mlflow_manager.log_params({
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'n_features': len(self.feature_cols)}
            )


            # Train XGboost
            xgb_model = self.train_xgboost(
                x_train, y_train, x_val, y_val,
                use_optuna=use_optuna
            )
            xgb_pred = xgb.model.predict(x_test)
            xgb_metrics = self.calculate_metrics(
                y_test, xgb_pred
            )

            self.mlflow_manager.log_metrics({f"xgboost_{k}": v for k, v in xgb_metrics.items()})
            self.mlflow_manager.log_model(xgb_model, "xgboost", 
                                         input_example=x_train.iloc[:5])

            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            logger.info(f"Top XGBoost features:\n{feature_importance.to_string()}")
            self.mlflow_manager.log_params({f"xgb_top_feature_{i}": f"{row['feature']} ({row['importance']:.4f})" 
                                           for i, (_, row) in enumerate(feature_importance.iterrows())})
            
            results['xgboost'] = {
                'model': xgb_model,
                'metrics': xgb_metrics,
                'predictions': xgb_pred
            }


            # Train LightGBM
            lgb_model = self.train_lightgbm(x_train, y_train, x_val, y_val, use_optuna)
            lgb_pred = lgb_model.predict(x_test)
            lgb_metrics = self.calculate_metrics(y_test, lgb_pred)
            
            self.mlflow_manager.log_metrics({f"lightgbm_{k}": v for k, v in lgb_metrics.items()})
            self.mlflow_manager.log_model(lgb_model, "lightgbm",
                                         input_example=x_train.iloc[:5])
            
            # Log feature importance for LightGBM
            lgb_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            logger.info(f"Top LightGBM features:\n{lgb_importance.to_string()}")
            
            results['lightgbm'] = {
                'model': lgb_model,
                'metrics': lgb_metrics,
                'predictions': lgb_pred
            }


            # Weighted ensemble based on individual model performance
            xgb_val_pred = xgb_model.predict(x_val)
            lgb_val_pred = lgb_model.predict(x_val)
                
            xgb_val_r2 = r2_score(y_val, xgb_val_pred)
            lgb_val_r2 = r2_score(y_val, lgb_val_pred)

            # Calculate weights based on R2 scores with minimum weight constraint
            # This prevents a poorly performing model from being completely ignored
            min_weight = 0.2
            xgb_weight = max(min_weight, xgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
            lgb_weight = max(min_weight, lgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
            
            # Normalize weights
            total_weight = xgb_weight + lgb_weight
            xgb_weight = xgb_weight / total_weight
            lgb_weight = lgb_weight / total_weight
            
            logger.info(f"Ensemble weights - XGBoost: {xgb_weight:.3f}, LightGBM: {lgb_weight:.3f}")

            ensemble_weights = {
                'xgboost': xgb_weight,
                'lightgbm' : lgb_weight
            }

            # Create the ensemble model object
            ensemble_models = {
                'xgboost': xgb_model,
                'lightgbm': lgb_model
            }


            ensemble_model = EnsembleModel(ensemble_models, ensemble_weights)

            self.models['ensemble'] = ensemble_models

            ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred)
            
            self.mlflow_manager.log_metrics({f"ensemble_{k}": v for k, v in ensemble_metrics.items()})
            self.mlflow_manager.log_model(ensemble_model, "ensemble", 
                                         input_example=x_train.iloc[:5])

            results['ensemble'] = {
                'model': ensemble_model,
                'metrics': ensemble_metrics,
                'predictions': ensemble_pred
            }






        except Exception as e:
            self.mlflow_manager.end_run(status="FAILED")
            raise e
        
        return results
        logger.info("Training all models")

