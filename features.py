import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_log_error


numerical_features = [
    "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
    "BsmtHalfBath", "FullBath", "HalfBath", "Bedroom", "Kitchen", "TotRmsAbvGrd", "Fireplaces",
    "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"
]

unordered_categorical_features = [
    "MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig",
    "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation",
    "Heating", "CentralAir", "Electrical", "GarageType", "MiscFeature", "SaleType", "SaleCondition"
]

ordered_features = [
    "LotShape", "Utilities", "LandSlope", "OverallQual", "OverallCond", "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "KitchenQual",
    "Functional", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive",
    "PoolQC", "Fence"
]



# preprocessing data
def initial_preprocess(df):
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



def impute(df):

    numerical_cols = [col for col in df.columns if df[col].dtype in ['int64',
                                                                     'float64']]

    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

    # Numerical features - fill missing with 0
    numerical_imputer = SimpleImputer(strategy='constant', fill_value=0)
    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

    # categorical features - fill missing with "None"
    categorical_imputer = SimpleImputer(strategy='constant', fill_value='None')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    return df


def encode_categorical(df, ordered_levels):

    # unordered categorical:
    for name in unordered_categorical_features:
        df[name] = df[name].astype('category')
        # add a 'None' category:
        if 'None' not in df[name].cat.categories:
            df[name] = df[name].cat.add_categories('None')
        df[name] = df[name].fillna('None')  # replace missing with 'None'

    # ordered
    for col, categories in ordered_levels.items():
        mapping = {label: i for i, label in enumerate(categories)}
        mapping[np.nan] = -1
        df[col] = df[col].map(lambda x: mapping.get(x, -1)).astype("int64")
    return df