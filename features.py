import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

# preprocessing data
def preprocess(df):
    numerical_cols = [col for col in df.columns if df[col].dtype in ['int64'
        , 'float64']]

    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

    numerical_transformer = SimpleImputer(strategy='constant', fill_value=0)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return preprocessor



def evaluate_model(preprocessor, X, y, model):
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Log-transform target for stability in RMSLE
    y_log = np.log1p(y)

    # Cross-validation RMSLE (RMSE on log target)
    cv_rmsle_scores = cross_val_score(
        pipe, X, y_log,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )
    cv_rmsle = -np.mean(cv_rmsle_scores)

    # Cross-validation MAE
    cv_mae_scores = cross_val_score(
        pipe, X, y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    cv_mae = -np.mean(cv_mae_scores)

    results = {
        "cv_rmsle": f"{cv_rmsle:.5f}",
        "cv_mae": f"{cv_mae:.5f}"
    }

    return results

