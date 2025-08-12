import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


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

ordered_cols_num = ['OverallQual', 'OverallCond', 'GarageQual', 'ExterQual',
    'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
            'FireplaceQu', 'GarageCond', 'PoolQC']

ordered_cols_str = ['Fence', 'PavedDrive', 'Utilities', 'CentralAir',
'Electrical', 'LotShape', 'LandSlope', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'Functional', 'GarageFinish']

five_levels = list(range(1, 6))
ten_levels = list(range(1, 11))

ordered_levels_num = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    'GarageQual': five_levels,
    'ExterQual': five_levels,
    'ExterCond': five_levels,
    'BsmtQual': five_levels,
    'BsmtCond': five_levels,
    'HeatingQC': five_levels,
    'KitchenQual': five_levels,
    'FireplaceQu': five_levels,
    'GarageCond': five_levels,
    'PoolQC': five_levels,
}
ordered_levels_str = {
    'Fence': ['MnWd', 'GdWo', "MnPrv", 'GdPrv'],
    'PavedDrive': ['N', 'P', 'Y'],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"]
}

# preprocessing data
def preprocess(df: pd.DataFrame, ordered_levels: dict) -> ColumnTransformer:
    df = df.copy()
    all_cols = df.columns

    ordered_cols = [c for c in ordered_levels.keys() if c in all_cols]
    for col in ordered_cols:
        df[col] = df[col].astype(str)
    num_cols = list(df.select_dtypes(include=["number"]).columns)

    # unordered = object columns minus ordered ones
    unordered_cols = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c not in ordered_cols
    ]

    # Build categories list for OrdinalEncoder in the exact column order
    ordered_categories_num = [ordered_levels_num[c] for c in ordered_cols_num]
    ordered_categories_str = [ordered_levels_str[c] for c in ordered_cols_str]

    ordered_pipe_num = Pipeline(steps=[
        # fill NaNs with a sentinel that isn’t in categories; unknowns map to -1
        ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
        ("encode", OrdinalEncoder(
            categories=ordered_categories_num,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64
        ))
    ])

    ordered_pipe_str = Pipeline(steps=[

        ("to_str", FunctionTransformer(lambda X: X.astype(str))),
        ("impute", SimpleImputer(strategy="constant", fill_value="None")),
        ("encode", OrdinalEncoder(
            categories=[ordered_levels_str[c] for c in ordered_cols_str],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])


    unordered_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="None")),
        # dense output so linear models work without surprises
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", StandardScaler())
])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("ord_num", ordered_pipe_num, ordered_cols_num),
            ("ord_str", ordered_pipe_str, ordered_cols_str),
            ("cat", unordered_pipe, unordered_cols),
        ],
        remainder="drop")

    return preprocessor



def evaluate_model(X, y, model):

    # Log-transform target for stability in RMSLE
    y_log = np.log1p(y)

    # Cross-validation RMSLE (RMSE on log target)
    cv_rmsle_scores = cross_val_score(
        model, X, y_log,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )
    cv_rmsle = -np.mean(cv_rmsle_scores)

    # Cross-validation MAE
    cv_mae_scores = cross_val_score(
        model, X, y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    cv_mae = -np.mean(cv_mae_scores)

    results = {
        "cv_rmsle": f"{cv_rmsle:.5f}",
        "cv_mae": f"{cv_mae:.5f}"
    }

    return results