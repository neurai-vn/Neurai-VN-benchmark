
"""

"""
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from collections import Counter
import numpy as np
import random
import os

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

CLASSIFIERS = {
    "cLR": LogisticRegression(random_state=SEED, max_iter=1000),
    "cRF": RandomForestClassifier(random_state=SEED),
    # "cXGB": XGBClassifier(random_state=SEED, eval_metric="logloss"),  ## TOO LONG
    "cFCNN": MLPClassifier(random_state=SEED, max_iter=500)
}

def cRG(y, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    y_pred = rng.choice(classes, size=len(y))
    return y_pred


def cBRG(y):
    majority_class = Counter(y).most_common(1)[0][0]
    return np.full_like(y, majority_class)


REGRESSORS = {
    "rLR": LinearRegression(),
    "rRF": RandomForestRegressor(random_state=SEED),
    # "rXGB": XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist", n_estimators=50), ## TOO LONG
    "rFCNN": MLPRegressor(random_state=SEED, max_iter=500)
}

def rRB(y, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(y.min(), y.max(), size=len(y))


def rAB(y):
    return np.full_like(y, np.mean(y), dtype=float)

