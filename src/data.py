import numpy as np
import pandas as pd

# loading data
def load_data():
    train = pd.read_csv("../data/raw/train.csv", index_col='Id')
    test = pd.read_csv("../data/raw/test.csv", index_col='Id')
    return train, test



