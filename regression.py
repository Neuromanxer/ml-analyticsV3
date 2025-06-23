import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from lightgbm import log_evaluation
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from typing import List, Tuple, Optional
from typing import List, Optional, Dict, Any, Optional
from preprocessing import preprocess_data

# XGBoost Parameters (Basic)
xgb_params = {
    "objective": "reg:squarederror",
    "n_estimators": 2000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 10,
    "random_state": 42,
}

# LightGBM Parameters (Basic)
lgb_params = {
    "objective": "regression",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": -1,  # No limit on tree depth
    "num_leaves": 31,  # Standard number of leaves
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

# CatBoost Parameters (Basic)
cat_params = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "border_count": 254,
    "loss_function": "RMSE",
    "random_state": 42,
    "verbose": 0,
}
class DataPreprocessor:
    """Handles consistent data preprocessing for training and test sets"""
    
    def __init__(self):
        self.categorical_mappings = {}
        self.fill_values = {}
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor on training data and transform"""
        df = df.copy()
        
        for col in df.columns:
            if col in ['ID']:  # Skip ID columns
                continue
                
            print(f"Processing column: {col} (dtype: {df[col].dtype})")
            
            if df[col].dtype == 'object':
                # Handle categorical data
                df[col] = df[col].fillna('MISSING').astype(str)
                unique_values = sorted(df[col].unique())
                self.categorical_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
                df[col] = df[col].map(self.categorical_mappings[col]).astype('int32')
                
            elif df[col].dtype in ['float64', 'float32']:
                # Handle float data
                if df[col].isnull().any():
                    fill_val = df[col].median()
                    if pd.isna(fill_val):
                        fill_val = 0.0
                    self.fill_values[col] = fill_val
                    df[col] = df[col].fillna(fill_val)
                df[col] = df[col].astype('float32')
                
            elif df[col].dtype in ['int64', 'int32']:
                # Handle integer data
                if df[col].isnull().any():
                    fill_val = df[col].median()
                    if pd.isna(fill_val):
                        fill_val = 0
                    self.fill_values[col] = int(fill_val)
                    df[col] = df[col].fillna(fill_val)
                df[col] = df[col].astype('int32')
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted preprocessor"""
        df = df.copy()
        
        for col in df.columns:
            if col in ['ID']:  # Skip ID columns
                continue
                
            if col in self.categorical_mappings:
                # Handle categorical data
                df[col] = df[col].fillna('MISSING').astype(str)
                
                # Map known categories, assign new categories to max_code + 1
                max_code = max(self.categorical_mappings[col].values())
                
                def safe_map(val):
                    if val in self.categorical_mappings[col]:
                        return self.categorical_mappings[col][val]
                    else:
                        # Assign new category a new code
                        new_code = max_code + 1
                        self.categorical_mappings[col][val] = new_code
                        return new_code
                
                df[col] = df[col].apply(safe_map).astype('int32')
                
            elif col in self.fill_values:
                # Use training fill values for consistency
                df[col] = df[col].fillna(self.fill_values[col])
                if df[col].dtype in ['float64', 'float32']:
                    df[col] = df[col].astype('float32')
                else:
                    df[col] = df[col].astype('int32')
            
            else:
                # Handle new columns not seen in training
                if df[col].dtype == 'object':
                    df[col] = pd.Categorical(df[col].fillna('MISSING')).codes.astype('int32')
                elif df[col].dtype in ['float64', 'float32']:
                    df[col] = df[col].fillna(0.0).astype('float32')
                elif df[col].dtype in ['int64', 'int32']:
                    df[col] = df[col].fillna(0).astype('int32')
        
        return df


class ModelTrainer:
    """Handles cross-validation training of multiple models"""
    
    def __init__(self, data: pd.DataFrame, n_splits: int = 5):
        self.data = data.copy()
        self.n_splits = n_splits
        self.preprocessor = DataPreprocessor()
    
    def train_model(self, model_params: dict, target_column: str, model_type: str) -> Tuple[List, np.ndarray]:
        """
        Train a model using K-Fold cross-validation
        
        Returns:
            models: List of trained models (one per fold)
            oof_predictions: Out-of-fold predictions for the entire dataset
        """
        print(f"\n=== Training {model_type} ===")
        
        # Prepare data
        df = self.data.copy()
        
        # Preprocess the data
        df_processed = self.preprocessor.fit_transform(df)
        
        X = df_processed.drop(columns=['ID', target_column], errors='ignore')
        y = df_processed[target_column]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Feature dtypes: {X.dtypes.value_counts()}")
        
        # Initialize cross-validation
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        models = []
        oof_predictions = np.zeros(len(y))
        
        # Train models for each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Training fold {fold + 1}/{self.n_splits}")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Initialize model
            if model_type == "LightGBM":
                model = LGBMRegressor(**model_params)
            elif model_type == "XGBoost":
                model = XGBRegressor(**model_params)
            elif model_type == "CatBoost":
                model = CatBoostRegressor(**model_params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train model
            if model_type == "LightGBM":
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                )
            elif model_type == "XGBoost":
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],

                )
            else:  # CatBoost
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],

                )
            
            # Make predictions on validation set
            val_preds = model.predict(X_val_fold)
            oof_predictions[val_idx] = val_preds
            
            models.append(model)
            
            # Calculate fold score
            fold_rmse = mean_squared_error(y_val_fold, val_preds, squared=False)
            print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")
        
        # Calculate overall CV score
        cv_rmse = mean_squared_error(y, oof_predictions, squared=False)
        cv_mae = mean_absolute_error(y, oof_predictions)
        cv_r2 = r2_score(y, oof_predictions)
        
        print(f"{model_type} CV Scores:")
        print(f"  RMSE: {cv_rmse:.4f}")
        print(f"  MAE: {cv_mae:.4f}")
        print(f"  R²: {cv_r2:.4f}")
        
        return models, oof_predictions


def align_datasets(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure train and test datasets have aligned columns and dtypes"""
    
    print("Aligning train and test datasets...")
    
    # Find missing and extra columns
    missing_cols = set(X_train.columns) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(X_train.columns)
    
    if missing_cols:
        print(f"Adding missing columns to test set: {missing_cols}")
        for col in missing_cols:
            X_test[col] = 0  # Fill with default value
    
    if extra_cols:
        print(f"Removing extra columns from test set: {extra_cols}")
        X_test = X_test.drop(columns=list(extra_cols))
    
    # Reorder columns to match training set
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # Ensure matching dtypes
    for col in X_train.columns:
        if X_test[col].dtype != X_train[col].dtype:
            print(f"Converting {col} from {X_test[col].dtype} to {X_train[col].dtype}")
            X_test[col] = X_test[col].astype(X_train[col].dtype)
    
    print("Dataset alignment completed")
    return X_train, X_test


def train_regression_models(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    target_column: str
) -> Dict[str, Any]:
    """
    Main function to train regression models
    
    Args:
        train_df: Training dataset
        test_df: Test dataset (optional, can be None)
        target_column: Name of the target column
    
    Returns:
        Dictionary containing results
    """
    
    print(f"Starting regression training...")
    print(f"Train dataset shape: {train_df.shape}")
    if test_df is not None:
        print(f"Test dataset shape: {test_df.shape}")
        has_test_target = target_column in test_df.columns
        print(f"Test dataset has target column: {has_test_target}")
    else:
        has_test_target = False
    
    # Prepare training data
    preprocessor = DataPreprocessor()
    train_processed = preprocessor.fit_transform(train_df)
    
    X_train = train_processed.drop(columns=['ID', target_column], errors='ignore')
    y_train = train_processed[target_column]
    
    # Prepare test data if available
    X_test, y_test = None, None
    if test_df is not None:
        test_processed = preprocessor.transform(test_df)
        
        if has_test_target:
            X_test = test_processed.drop(columns=['ID', target_column], errors='ignore')
            y_test = test_processed[target_column]
        else:
            X_test = test_processed.drop(columns=['ID'], errors='ignore')
        
        # Align datasets
        X_train, X_test = align_datasets(X_train, X_test)
    
    # Train models using cross-validation
    trainer = ModelTrainer(train_df, n_splits=5)
    
    print("\n" + "="*50)
    print("CROSS-VALIDATION TRAINING")
    print("="*50)
    
    lgb_models, lgb_oof = trainer.train_model(lgb_params, target_column, "LightGBM")
    xgb_models, xgb_oof = trainer.train_model(xgb_params, target_column, "XGBoost")
    cat_models, cat_oof = trainer.train_model(cat_params, target_column, "CatBoost")
    
    # Calculate CV scores
    cv_scores = {
        "LightGBM": {
            "rmse": mean_squared_error(y_train, lgb_oof, squared=False),
            "mae": mean_absolute_error(y_train, lgb_oof),
            "r2": r2_score(y_train, lgb_oof)
        },
        "XGBoost": {
            "rmse": mean_squared_error(y_train, xgb_oof, squared=False),
            "mae": mean_absolute_error(y_train, xgb_oof),
            "r2": r2_score(y_train, xgb_oof)
        },
        "CatBoost": {
            "rmse": mean_squared_error(y_train, cat_oof, squared=False),
            "mae": mean_absolute_error(y_train, cat_oof),
            "r2": r2_score(y_train, cat_oof)
        }
    }
    
    # Find best model
    best_model_name = min(cv_scores, key=lambda m: cv_scores[m]["rmse"])
    best_models = {
        "LightGBM": lgb_models,
        "XGBoost": xgb_models,
        "CatBoost": cat_models
    }[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print("="*50)
    
    # Make predictions on test set if available
    test_predictions = None
    test_scores = None
    
    if X_test is not None:
        print("Making predictions on test set...")
        # Use ensemble of all folds for prediction
        test_predictions = np.mean([model.predict(X_test) for model in best_models], axis=0)
        
        if has_test_target and y_test is not None:
            test_scores = {
                "rmse": mean_squared_error(y_test, test_predictions, squared=False),
                "mae": mean_absolute_error(y_test, test_predictions),
                "r2": r2_score(y_test, test_predictions)
            }
            print(f"Test Scores: {test_scores}")
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    
    if test_df is not None and has_test_target:
        # Combine train and test for final training
        full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    else:
        # Use only training data
        full_df = train_df
    
    full_processed = preprocessor.fit_transform(full_df)
    X_full = full_processed.drop(columns=['ID', target_column], errors='ignore')
    y_full = full_processed[target_column]
    
    # Train final model (use first model from best performing algorithm)
    if best_model_name == "LightGBM":
        final_model = LGBMRegressor(**lgb_params)
    elif best_model_name == "XGBoost":
        final_model = XGBRegressor(**xgb_params)
    else:  # CatBoost
        final_model = CatBoostRegressor(**cat_params)
    
    final_model.fit(X_full, y_full)
    
    # Prepare results
    results = {
        "best_model_name": best_model_name,
        "cv_scores": cv_scores,
        "final_model": final_model,
        "preprocessor": preprocessor,
        "test_predictions": test_predictions.tolist() if test_predictions is not None else None,
        "test_scores": test_scores,
        "feature_names": X_train.columns.tolist()
    }
    
    return results
# Improved visualization generation in the regression endpoint
def generate_visualizations_improved(results, train_df, user_dir, user_id):
    """Generate visualizations with proper error handling"""
    try:
        # Create proper sample data for visualization
        X_sample = create_sample_data_for_visualization(
            results["feature_names"], 
            train_df, 
            results["preprocessor"]
        )
        
        print(f"Created sample data with shape: {X_sample.shape}")
        
        fi_b64, det_b64, imp_df = plot_feature_importance(
            results["final_model"],
            X_sample,  # Use proper sample data instead of empty DataFrame
            model_type='regressor',
            output_dir=user_dir,
            save_filename=f"{user_id}_feature_importance.png",
            return_base64=True
        )
        
        return fi_b64, det_b64, imp_df
        
    except Exception as viz_error:
        print(f"Visualization error: {viz_error}")
        import traceback
        traceback.print_exc()
        
        # Try fallback with just feature importance without SHAP
        try:
            # Get basic feature importance
            if hasattr(results["final_model"], 'feature_importances_'):
                importances = results["final_model"].feature_importances_
                imp_df = pd.DataFrame({
                    'Feature': results["feature_names"],
                    'Importance': importances * 100  # Convert to percentage
                }).sort_values('Importance', ascending=False)
                
                return None, None, imp_df
            else:
                return None, None, pd.DataFrame()
                
        except Exception as fallback_error:
            print(f"Fallback visualization also failed: {fallback_error}")
            return None, None, pd.DataFrame()
