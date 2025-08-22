
import pandas as pd
from category_encoders import MEstimateEncoder
from src.target_encoding import CrossFoldEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



def load_data():
    train = pd.read_csv("../data/raw/train.csv", index_col='Id')
    test = pd.read_csv("../data/raw/test.csv", index_col='Id')
    return train, test


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
    "Heating", "Electrical", "GarageType", "MiscFeature", "SaleType", "SaleCondition"
]

ordered_features = [
    "LotShape", "Utilities", "LandSlope", "OverallQual", "OverallCond", "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "KitchenQual",
    "Functional", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive",
    "PoolQC", "Fence", "HeatingQC"
]

binary_cols = ["CentralAir"]


def preprocess():
    df_train, df_test = load_data()

    # Merge so we can process them together
    df = pd.concat([df_train, df_test])

    # preprocessing functions
    df = clean(df)
    df = encode(df)
    df = impute(df)

    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :]

    return df_train, df_test



def clean(df):

    out = df.copy()

    # 1. make categorical
    for col in ["MSSubClass"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    # 2. numeric coercion for a few columns that sometimes import as 'object'
    for col in ["LotFrontage", "MasVnrArea", "GarageYrBlt"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def encode(df):
    out = df.copy()

    # 1. identify target and row groups
    target = "SalePrice"
    train_mask = out[target].notna()
    test_mask = ~train_mask

    train_idx = out.index[train_mask]
    test_idx = out.index[test_mask]
    y = out.loc[train_idx, target]

    # 2. ordinal mapping
    qual_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    expo_map = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}

    bsmtfin_map = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

    func_map = {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}

    garagefin_map = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}

    paved_map = {"N": 0, "P": 1, "Y": 2}

    fence_map = {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}

    lotshape_map = {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}

    utilities_map = {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}

    landslope_map = {"Sev": 1, "Mod": 2, "Gtl": 3}

    qual_like = {"ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "KitchenQual",
                 "FireplaceQu", "GarageQual", "GarageCond", "PoolQC", "HeatingQC"}
    expo_like = {"BsmtExposure"}
    bsmtfin_like = {"BsmtFinType1", "BsmtFinType2"}
    func_like = {"Functional"}
    garagefin_like = {"GarageFinish"}
    paved_like = {"PavedDrive"}
    fence_like = {"Fence"}
    lotshape_like = {"LotShape"}
    utilities_like = {"Utilities"}
    landslope_like = {"LandSlope"}

    # Apply mappings (skip numeric ordinals OverallQual/OverallCond)
    for col in ordered_features:
        if col not in out.columns:
            continue
        if col in {"OverallQual", "OverallCond"}:
            # these are already numeric 1..10; leave as-is
            continue

        if col in qual_like:
            out[col] = out[col].map(qual_map).fillna(0)
        elif col in expo_like:
            out[col] = out[col].map(expo_map).fillna(0)
        elif col in bsmtfin_like:
            out[col] = out[col].map(bsmtfin_map).fillna(0)
        elif col in func_like:
            out[col] = out[col].map(func_map).fillna(0)
        elif col in garagefin_like:
            out[col] = out[col].map(garagefin_map).fillna(0)
        elif col in paved_like:
            out[col] = out[col].map(paved_map).fillna(0)
        elif col in fence_like:
            out[col] = out[col].map(fence_map).fillna(0)
        elif col in lotshape_like:
            out[col] = out[col].map(lotshape_map).fillna(0)
        elif col in utilities_like:
            out[col] = out[col].map(utilities_map).fillna(0)
        elif col in landslope_like:
            out[col] = out[col].map(landslope_map).fillna(0)
        else:
            # If an ordered feature isn’t covered above, leave it unchanged.
            pass

    # 3. Any category with count < 10 becomes "Other" (train and test).
    for col in unordered_categorical_features:
        if col not in out.columns:
            continue

        value_counts = out.loc[train_idx, col].value_counts(dropna=True)
        keep = set(value_counts[value_counts >= 10].index)

        # If column is categorical, add "Other" to categories first
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].cat.add_categories(["Other"])
            mask = out[col].notna() & ~out[col].isin(keep)
            out.loc[mask, col] = "Other"
            out[col] = out[col].cat.remove_unused_categories()
        else:
            mask = out[col].notna() & ~out[col].isin(keep)
            out.loc[mask, col] = "Other"

    # 4. One-Hot Encode low-cardinality nominals
    low_card_cols = []
    for col in unordered_categorical_features:
        if col in out.columns:
            n_unique_train = out.loc[train_idx, col].nunique(dropna=True)
            if n_unique_train <= 12:
                low_card_cols.append(col)

    if low_card_cols:
        dummies = pd.get_dummies(out[low_card_cols], prefix=low_card_cols, prefix_sep="__", dtype="uint8")
        out = pd.concat([out.drop(columns=low_card_cols), dummies], axis=1)

    # 5. Target-encode high-cardinality nominals
    high_card_cols = [c for c in unordered_categorical_features if c in out.columns and c not in low_card_cols]

    if high_card_cols:
        # Train/test slices for just these columns
        X_tr = out.loc[train_idx, high_card_cols]
        X_te = out.loc[test_idx, high_card_cols]
        y_tr = out.loc[train_idx, target]

        # Cross-fold target encoder (smoothed)
        encoder = CrossFoldEncoder(
            MEstimateEncoder,
            m=20,
            handle_unknown="value",
            handle_missing="value"
        )

        te_tr = encoder.fit_transform(X_tr, y_tr, cols=high_card_cols)  # train rows only
        te_te = encoder.transform(X_te)  # test rows

        # write back into the combined frame, aligned by index
        te_cols = te_tr.columns.tolist()
        out.loc[train_idx, te_cols] = te_tr
        out.loc[test_mask, te_cols] = te_te

        # drop the original high-card categorical columns
        out = out.drop(columns=high_card_cols)

    # 6. Encode binary columns
    binary_maps = {
        "CentralAir": {"N": 0, "Y": 1},
        # add more binary cols here if needed
    }
    for col, mapping in binary_maps.items():
        if col in binary_cols:
            out[col] = out[col].map(mapping).astype("uint8")

    # 7. Final Checks

    # Type tidy-up for encoded features
    ohe_cols = [c for c in out.columns if "__" in c]  # one-hot dummies
    te_cols = [c for c in out.columns if c.endswith("_encoded")]  # target-encoded

    if ohe_cols:
        out[ohe_cols] = out[ohe_cols].astype("uint8")
    if te_cols:
        out[te_cols] = out[te_cols].astype("float32")

    # Sanity: encoding columns should not contain NaN
    enc_cols = ohe_cols + te_cols
    if enc_cols:
        has_nan = out[enc_cols].isna().any().any()
        if has_nan:
            missing_in = [c for c in enc_cols if out[c].isna().any()]
            raise ValueError(f"Encoding produced NaNs in columns: {missing_in}")

    # confirm all model features are numeric
    skip = {"SalePrice", "Id"}
    non_numeric = [c for c in out.columns
                   if c not in skip and not pd.api.types.is_numeric_dtype(out[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns remain after encoding: {non_numeric}")

    return out


def impute(df):
    out = df.copy()

    # 1. numeric imputation: split by semantics
    zero_fill = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                 "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
                 "PoolArea", "MasVnrArea", "Fireplaces", "GarageYrBlt"]
    num_cols = out.select_dtypes(include="number").columns

    for col in num_cols:
        if out[col].isna().any():
            if col in zero_fill:
                out[col] = out[col].fillna(0)
            else:
                out[col] = out[col].fillna(out[col].median())

    # 2. categorical imputation is usually unnecessary after encode
    # (kept here as a no-op safety net)
    cat_cols = out.select_dtypes(include=["category", "object"]).columns
    for col in cat_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna("Unknown")

    return out


def evaluate_model(X, y, model=None):
    if model is None:
        model = XGBRegressor(tree_method="hist", random_state=42)

    # Log-transform target for RMSLE
    log_y = np.log(y)

    # 5-fold CV, scoring = neg MSE
    scores = cross_val_score(
        model, X, log_y,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    rmse = np.sqrt(-scores.mean())

    # MAE on original target scale
    mae_scores = cross_val_score(
        model, X, y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    mae = -mae_scores.mean()

    print(f"RMSLE: {rmse:.5f}")
    print(f"MAE: {mae:.5f}")

    return rmse, mae


def get_mi(X, y):
    X = X.copy()
    # mark discrete features
    discrete = [X[c].nunique() <= 12 for c in X.columns]
    mi = mutual_info_regression(X, y, discrete_features=discrete, random_state=0)
    return pd.Series(mi, index=X.columns)


def plot_mi(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def drop_zeromi(df, mi_scores):
    return df.loc[:, mi_scores > 0.0]


def interaction_features(df):
    # TotalSF = above-ground + basement
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]

    # TotalBaths (full + half*0.5, incl. basement)
    df["TotalBaths"] = (
            df["FullBath"] + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )

    # porchSF
    df["PorchSF"] = (
            df["OpenPorchSF"] + df["EnclosedPorch"]
            + df["3SsnPorch"] + df["ScreenPorch"]
    )


    # ratio
    rooms = np.maximum(df["TotRmsAbvGrd"].astype(float), 1.0)
    df["BathsPerRoom"] = df["TotalBaths"] / rooms

    df["LivLotRatio"] = df["GrLivArea"] / df["LotArea"]
    df["Spaciousness"] = (df["1stFlrSF"] + df["2ndFlrSF"]) / df["TotRmsAbvGrd"]

    # counts
    df["PorchTypes"] = df[[
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
    ]].gt(0.0).sum(axis=1)

    return df



def kmeans_cluster(df, features, k=5):

    col = "Clusters"
    out = df.copy()

    # scale (fit on train only)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(out[features])

    # kmeans (fit on train only)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)

    # add labels
    out[col] = km.predict(X_scaled).astype("int16")

    return out