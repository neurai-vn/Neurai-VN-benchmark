
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


#===========================#
# Function 
#===========================#
CLASSIFIERS = {
    "cLR": LogisticRegression(random_state=SEED, max_iter=1000),
    "cRF": RandomForestClassifier(random_state=SEED),
    "cXGB": XGBClassifier(random_state=SEED, eval_metric="logloss", n_estimators=200),
    "cFCNN": MLPClassifier(random_state=SEED, max_iter=500)
}



#===========================#
# Function 
#===========================#
REGRESSORS = {
    "rLR": LinearRegression(),
    "rRF": RandomForestRegressor(random_state=SEED),
    "rXGB": XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist", n_estimators=200),
    "rFCNN": MLPRegressor(random_state=SEED, max_iter=500)
}

