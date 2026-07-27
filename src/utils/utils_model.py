from time import time

from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.pipeline import Pipeline

from src.benchmarkML.models import CLASSIFIERS, REGRESSORS
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    mean_absolute_error, 
    r2_score
)
from sklearn.base import clone
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


GLOBAL_SEEDS = 42


##########################
def preprocess_feature(x: np.array):
    """
    preprocess the feature with imputation and scaling. 
    Return preprocesser for test set transform
    """

    # Impute missing values with mean imputation
    imputer = SimpleImputer(strategy="mean")
    x = imputer.fit_transform(x)

    # Scale the feature with min-max scaler
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    return x, imputer, scaler


##########################
def _get_model(model_type:str):
    """
    get ML models
    """
    if model_type == 'classification':
        return CLASSIFIERS
    elif model_type == 'regression':
        return REGRESSORS
    else:
        raise NotImplementedError()
    


##########################
def _get_score(model_type, y_test, y_pred):
    """
    get prediction scores
    """
    if model_type == 'classification':
        # return f1_score(y_test, y_pred, average='weighted')
        return f1_score(y_test, y_pred, average='macro')
    elif model_type == 'regression':
        return mean_absolute_error(y_test, y_pred)
    else:
        raise NotImplementedError()





##########################
def pipeline_heldout(
    x_train, 
    y_train,
    x_test,  
    y_test,  
    groups_test,
    model_type="classification"
):
    """
    Train on full training set and evaluate on held-out test set
    using subject-level aggregation.
    """

    func_model = _get_model(model_type)
    df_result = pd.DataFrame()
    j = 0

    for model_name, clf in func_model.items():

        # ---- Train on FULL training data ----
        clf.fit(x_train, y_train)
        # ---- Predict on held-out test ----
        y_pred = clf.predict(x_test)
        # ---- Sample-level score ----
        score = _get_score(model_type, y_test, y_pred)
        
        # ---- Subject-level metrics ----
        if model_type == "classification":
            sb_metrics = subject_level_classification_metrics(
                groups=groups_test,
                test_idx=np.arange(len(y_test)),
                y_test=y_test,
                y_pred=y_pred
            )
        elif model_type == "regression":
            sb_metrics = subject_level_regression_metrics(
                groups=groups_test,
                test_idx=np.arange(len(y_test)),
                y_test=y_test,
                y_pred=y_pred
            )
        # ---- Store results ----
        df_result.loc[j, "split"] = "held-out-test"
        df_result.loc[j, "model_name"] = model_name
        df_result.loc[j, "score"] = score
  
        for key, val in sb_metrics.items():
            df_result.loc[j, f"subject_{key}"] = val
        j += 1

    return df_result


##########################
def _get_kfold(model_type:str, K:int):
    """
    get kfold strategy
    """
    if model_type == 'classification':
        return StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=42)
    elif model_type == 'regression':
        return GroupKFold(n_splits=K)
    else:
        raise NotImplementedError()
    


##########################
def build_pipeline(
    model,
    model_type="classification",
    k_features=20
):
    """
    preprocessing + feature selection + model
    """

    if model_type == "classification":
        selector = SelectKBest(
            score_func=f_classif,
            k=k_features
        )
    else:
        selector = SelectKBest(
            score_func=f_regression,
            k=k_features
        )

    pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="mean")
        ),
        (
            "scaler",
            MinMaxScaler()
        ),
        (
            "selector",
            selector
        ),
        (
            "model",
            clone(model)
        )
    ])

    return pipe




##########################
def pipeline_kfold(x, y, groups, model_type, n_splits=5):
    """
    -classifier baselines -> F1 score tasks 
    -regression baselines -> mean absolute error (MAE)
    -stratified group 5-fold cross-validation (SG5FCV) - 
    -split data into non-overlapping participant training and test sets
    """
    # get model
    
    func_model = _get_model(model_type)
    # Cross-validation
    kf = _get_kfold(model_type, K=n_splits)

    df_result = pd.DataFrame()
    j = 0
    for model_name, _clf in func_model.items():
        score_fold = []
        for i, (train_idx, val_idx) in enumerate(kf.split(x, y, groups)):

            # x_train, x_val = x[train_idx], x[val_idx]
            # y_train, y_val = y[train_idx], y[val_idx]
            if isinstance(x, pd.DataFrame):
                x_train = x.iloc[train_idx]
                x_val = x.iloc[val_idx]
            else:
                x_train = x[train_idx]
                x_val = x[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]

            
            ##========= CONVENTIONAL WAY=========##
            ## impute + normalize
            x_train, imputer, scaler = preprocess_feature(x_train)
            x_val = imputer.transform(x_val)
            x_val = scaler.transform(x_val)

            K = min(max(20, int(0.5 * x_train.shape[1])), x_train.shape[1])            
            selector = SelectKBest(
                score_func=f_classif if model_type == "classification" else f_regression,
                k=K
            )
            x_train = selector.fit_transform(x_train, y_train)
            x_val = selector.transform(x_val)

            ## TRAIN & EVALUATION
            clf = clone(_clf)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_val)

            # ##========= PIPELINE WAY =========##
            # clf = clone(_clf)
            # pipe = build_pipeline(clf, model_type=model_type, k_features=20)
            # pipe.fit(x_train, y_train)
            # y_pred = pipe.predict(x_val)

            
            score = _get_score(model_type, y_val, y_pred)

            ## subject-level
            if model_type == "classification":
                sb_metrics = subject_level_classification_metrics(groups, val_idx, y_val, y_pred)
            elif model_type == "regression":
                sb_metrics = subject_level_regression_metrics(groups, val_idx, y_val, y_pred)
            #
            df_result.loc[j, "fold"] = f"fold-{i+1}"
            df_result.loc[j, "model_name"] = model_name
            df_result.loc[j, "score"] = score
            for key in sb_metrics.keys():
                df_result.loc[j, f"subject_{key}"] = sb_metrics[key]
            j += 1
            # print(f"\n>>>>Model: {model_name} | Fold: {i+1} | Score: {score:.4f}")
            # print(df_result)

    return df_result





##########################
def pipeline_repeated_kfold(
    x,
    y,
    groups,
    model_type,
    n_splits=5,
    n_repeats=10,
):
    
    func_model = _get_model(model_type)

    rows = []

    for repeat_id in range(n_repeats):

        if model_type == "classification":
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=repeat_id
            )
        else:
            cv = GroupKFold(n_splits=n_splits)

        for model_name, base_model in func_model.items():
            for fold_id, (train_idx, val_idx) in enumerate(cv.split(x, y, groups)):
                t_start = time()
                
                x_train = x[train_idx]
                x_val = x[val_idx]

                y_train = y[train_idx]
                y_val = y[val_idx]

                ##========= CONVENTIONAL WAY=========##
                # preprocessing
                x_train, imputer, scaler = preprocess_feature(x_train)
                x_val = imputer.transform(x_val)
                x_val = scaler.transform(x_val)
                # train
                clf = clone(base_model)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_val)


                # ##========= PIPELINE WAY =========##
                # clf = clone(base_model)
                # pipe = build_pipeline(clf, model_type=model_type, k_features=20)
                # pipe.fit(x_train, y_train)
                # y_pred = pipe.predict(x_val)




                score = _get_score(
                    model_type,
                    y_val,
                    y_pred
                )

                # subject-level
                if model_type == "classification":
                    sb_metrics = subject_level_classification_metrics(
                        groups,
                        val_idx,
                        y_val,
                        y_pred
                    )
                else:
                    sb_metrics = subject_level_regression_metrics(
                        groups,
                        val_idx,
                        y_val,
                        y_pred
                    )

                row = {
                    "repeat": repeat_id + 1,
                    "fold": fold_id + 1,
                    "model_name": model_name,
                    "score": score,
                }

                for k, v in sb_metrics.items():
                    row[f"subject_{k}"] = v

                rows.append(row)
                print(f">>>> Repeat: {repeat_id+1} | Model: {model_name} | Fold: {fold_id+1} | Score: {score:.4f} | Time: {time()-t_start:.2f} sec")

    return pd.DataFrame(rows)





##########################
def subject_level_classification_metrics(groups, test_idx, y_test, y_pred, average='macro'):
    """
    Compute subject-level metrics by aggregating predictions using majority voting.
    Parameters:
        groups     -- array of subject IDs for each sample
        test_idx   -- test indices
        y_test     -- true labels
        y_pred     -- predicted labels
        average    -- averaging method for multi-class metrics ('macro', 'weighted', etc.)
    Returns:
        A dictionary of metrics: accuracy, f1, precision, recall
    """
    
    subject_y_true = defaultdict(list)
    subject_y_pred = defaultdict(list)

    for j in range(len(y_test)):
        sb = groups[test_idx[j]]
        subject_y_true[sb].append(y_test[j])
        subject_y_pred[sb].append(y_pred[j])

    final_true = []
    final_pred = []

    for sb in subject_y_true:
        true_labels = subject_y_true[sb]
        pred_labels = subject_y_pred[sb]

        # Use first true label (assumed consistent per subject)
        true_label = true_labels[0]

        # Majority vote for predicted label
        pred_label = Counter(pred_labels).most_common(1)[0][0]

        final_true.append(true_label)
        final_pred.append(pred_label)

    return {
        'accuracy': accuracy_score(final_true, final_pred),
        'f1': f1_score(final_true, final_pred, average=average),
        'precision': precision_score(final_true, final_pred, average=average),
        'recall': recall_score(final_true, final_pred, average=average)
    }

    

##########################
def subject_level_regression_metrics(groups, test_idx, y_test, y_pred):
    """
    Compute subject-level regression metrics by averaging predictions per subject.
    
    Parameters:
        groups     -- array of subject IDs for each sample
        test_idx   -- test indices
        y_test     -- true continuous values
        y_pred     -- predicted continuous values

    Returns:
        A dictionary of metrics: MAE, RMSE, R2
    """
    
    subject_y_true = defaultdict(list)
    subject_y_pred = defaultdict(list)

    for j in range(len(y_test)):
        sb = groups[test_idx[j]]
        subject_y_true[sb].append(y_test[j])
        subject_y_pred[sb].append(y_pred[j])

    final_true = []
    final_pred = []

    for sb in subject_y_true:
        true_vals = subject_y_true[sb]
        pred_vals = subject_y_pred[sb]

        # Average over the subject's samples
        final_true.append(np.mean(true_vals))
        final_pred.append(np.mean(pred_vals))

    # Compute metrics on subject-level aggregated values
    mae = mean_absolute_error(final_true, final_pred)
    r2 = r2_score(final_true, final_pred)

    return {
        'mae': mae,
        'r2': r2
    }