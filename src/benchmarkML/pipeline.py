import os
import pandas as pd
import numpy as np
from typing import List, Dict
from src.utils.utils_model import (
    preprocess_feature,
    pipeline_repeated_kfold,
    pipeline_kfold,
    pipeline_heldout
)
from src.utils import utils_data
from src.utils.logger import get_logger
# others
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
##
logger = get_logger(__name__)


IDENTIFIER = "public_id"
COLUMN_LABEL = "label"

#===========================#
# Function 
#===========================#
class Baselineneuraivn:
    def __init__(
        self, 
        verbose: bool = False,
        list_streams: List[str] = None,
        **kwargs
    ):
        self.kwargs = kwargs
        self.list_streams = list_streams
        self.verbose = verbose
        self.modeling_tasks = [
            'hc-clinical', 
            'hc-dep', 
            'hc-anx', 
            'dep-anx'
        ]
        self.d_group_features = {
            "Wm": [
                "ATS", 
                "HRTS", 
                "AZMTS"
            ],
            "Ws": [
                "SLEEP", 
                "HRV",
                "SPO2",
                "BREATHINGRATE",
                "SKINTEMP"
            ],
            "Sd": [
                "DAILYSURVEY", 
                "MOODLOG"
            ],
            "P": [
                "APPSTATE",
                "ACC",
                "GYRO",
                "NETWORK",
                "BATTERY"
            ]
        }

    #------------------------#
    def get_data_by_task(
        self,
        df: pd.DataFrame,
        task: str,
        list_features: List[str]
    ):

        if task == "4class":
            # Four-class classification
            label_map = {
                "Control": 0,
                "Depression": 1,
                "Anxiety": 2,
                "OTHERS": 3,
            }
            df = df[df["label"].isin(label_map)]

        elif task == "3class":
            # Three-class classification (Control, Depression, Anxiety)
            label_map = {
                "Control": 0,
                "Depression": 1,
                "Anxiety": 2,
            }
            df = df[df["label"].isin(label_map)]


        elif task == "hc-clinical":
            # Clinical vs Healthy Control
            # Clinical = Depression + Anxiety + Others
            df = df[df["label"].isin(["Control", "Depression", "Anxiety", "OTHERS"])].copy()

            df["label"] = df["label"].replace({
                "Control": "Healthy",
                "Depression": "Clinical",
                "Anxiety": "Clinical",
                "OTHERS": "Clinical",
            })

            label_map = {
                "Healthy": 0,
                "Clinical": 1,
            }


        elif task == "hc-dep":
            # Depression vs Healthy Control
            label_map = {
                "Control": 0,
                "Depression": 1,
            }
            df = df[df["label"].isin(label_map)]

        elif task == "hc-anx":
            # Anxiety vs Healthy Control
            label_map = {
                "Control": 0,
                "Anxiety": 1,
            }
            df = df[df["label"].isin(label_map)]

        elif task == "dep-anx":
            # Depression vs Anxiety
            label_map = {
                "Depression": 0,
                "Anxiety": 1,
            }
            df = df[df["label"].isin(label_map)]

        else:
            raise ValueError(f"Unknown task: {task}")

        y = df["label"].map(label_map).to_numpy()
        x = df[list_features].to_numpy()
        groups = df[IDENTIFIER].values
        model_type = "classification"

        print(f"Task: {task}")
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        print("Class distribution:", dict(zip(*np.unique(y, return_counts=True))))

        return x, y, groups, model_type

    #------------------------#
    def _get_feature_original(self, subject:str):
        """
        Re-load the processed data of 1 subject
        """
        # print("\n<_get_feature_original>: subject: ", subject)
        return utils_data.load(
            os.path.join(
                self.kwargs['DIR_PROCESSED'], 
                'feature_original', 
                f'{subject}.pkl'
            )
        )
    

    #------------------------#
    def _get_feature_list(self, list_cols: List[str], group_feature: str = None) -> List[str]:
        """
        Get a list of feature columns if they are in list_streams (e.g., APPSTATE)
        """
        list_features = []
        for s in self.list_streams:
            list_feature = [col for col in list_cols if "#" in col and col.startswith(s)]
            list_features.extend(list_feature)

        ## Filter features based on group_feature
        if group_feature is not None: # for example, "Wm" or "Wm+Ws"
            if "+" in group_feature:
                groups = group_feature.split("+")
            else:
                groups = [group_feature]

            # Get the list of streams for the specified groups
            streams_to_include = []
            for g in groups:
                streams_to_include.extend(self.d_group_features.get(g, []))

            # Filter list_features to include only those that start with the specified streams
            list_features = [f for f in list_features if any(f.startswith(stream) for stream in streams_to_include)]

        return list_features
    
    #------------------------#
    def _get_feature_cols(self, list_cols: List[str]):
        pass

    #------------------------#
    def concat_modality_single_subject(self, data:Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        concatenate all features, dropNa

        So we use AZMTS, ATS, HRTS, APPSTATE sensor.
        The raw sensor of APPSTATE may have additional day, 
        but we will omit it, only cover those data within the start-end of HRTS

        """
        # concat
        df = []
        for (stream, v) in data.items():
            # if stream.startswith('feature'):
            df.append(v)
        df = pd.concat(df, axis=1)

        # Extract date part from the index and convert to datetime
        df[IDENTIFIER] = df.index.str.split('_').str[0]
        df['date'] = pd.to_datetime(df.index.str.split('_').str[1])
        df['day'] = (df['date'] - df['date'].min()).dt.days + 1
        df.reset_index(drop=True, inplace=True)

        ## sort date?
        df = df.sort_values(by=['date', IDENTIFIER]).reset_index(drop=True)
        
        # print("<concat_modality_single_subject> df:\n ", df)
        # print("<concat_modality_single_subject> df.shape: ", df.shape)
        days = df['date'].nunique()
        # print("<concat_modality_single_subject> Extract {days} days".format(days=days))
        return df


    #------------------------#
    def extract_df_feature_all_subject(self, df_label: pd.DataFrame):
        """
        Returns a dataframe feature for all subjects.
        """
        # get list of subjects
        list_subjects = list(df_label[IDENTIFIER])

        # concatenate all feature dataframes.
        df = pd.concat(
            [self.concat_modality_single_subject(
                self._get_feature_original(sb)
            ) \
                for sb in list_subjects],
            axis=0,
        ).reset_index(drop=True)

        # add demographic and annoation
        for i in df.index:
            info = df_label[df_label[IDENTIFIER]==df.loc[i, IDENTIFIER]]
            for c in info.columns[1:]:
                df.loc[i, c] = info[c].values[0]

        return df





#===========================#
# Function 
#===========================#
def run_ml( 
    task: str = "hc-dep",
    list_streams: List[str] = None,
    path_save: str = f"./assets/results/benchmarkML/baseline_neuraivn",
    pipeline_type: str = "kfold",
    n_splits: int = 5,
    n_repeats: int = 3,
    group_feature: str = None,
    **kwargs
):
    """
    main flow
    """
    print("\n======= RUN BENCHMARK ML PIPELINE ======")

    ## Step 0: get df_label_dev, df_label_test
    df_label_dev = pd.read_csv(os.path.join(kwargs['DIR_PROCESSED'], "label_clean_dev.csv"))
    df_label_test = pd.read_csv(os.path.join(kwargs['DIR_PROCESSED'], "label_clean_test.csv"))
    logger.debug(f"df_label_dev: {df_label_dev.shape} | df_label_test: {df_label_test.shape}")
    logger.debug(f"df_label_dev columns: {df_label_dev.columns}")

    ## Check label distribution
    logger.debug(f"label distribution in dev: {df_label_dev[COLUMN_LABEL].value_counts()}")
    logger.debug(f"label distribution in test: {df_label_test[COLUMN_LABEL].value_counts()}")

    ## Check subject
    list_subject_dev = df_label_dev[IDENTIFIER].unique().tolist()
    list_subject_test = df_label_test[IDENTIFIER].unique().tolist()
    logger.info(f"\n>>> Subjects in DEV: {list_subject_dev} (total = {len(list_subject_dev)})")
    logger.info(f"Subjects in TEST: {list_subject_test} (total = {len(list_subject_test)})")


    ## Call class & extract data
    dataClass_baseline = Baselineneuraivn(list_streams=list_streams, **kwargs)
    
    data_dev = dataClass_baseline.extract_df_feature_all_subject(df_label_dev)
    data_test = dataClass_baseline.extract_df_feature_all_subject(df_label_test)
    valid_dev = set(zip(data_dev[IDENTIFIER], data_dev.date))
    valid_test = set(zip(data_test[IDENTIFIER], data_test.date))

    
    ## check
    print(f"\n>>> data_dev: {data_dev.shape} | data_test: {data_test.shape}")
    print("\n>>> data_dev columns: ", data_dev.columns)
        
    ## ====Step 2: get feature columns for baseline performance.
    list_features = dataClass_baseline._get_feature_list(data_dev.columns, group_feature=group_feature)
    print(f"\n>>> group_feature: {group_feature} | list_features: {list_features} (total = {len(list_features)})")


    # ==========================================================
    # STEP 3: PREPARE DATA
    # ==========================================================
    print("\n\n==== Development Set ====")
    print(data_dev)
    x_dev, y_dev, groups_dev, model_type = dataClass_baseline.get_data_by_task(
        data_dev,
        task,
        list_features
    )


    print("\n\n==== Held-out Test Set ====")
    x_test, y_test, groups_test, _ = dataClass_baseline.get_data_by_task(
        data_test,
        task,
        list_features
    )

    metric = (
        "subject_f1"
        if model_type == "classification"
        else "subject_mae"
    )

    print(f"\n>>> DEV shape: {x_dev.shape}")
    print(f">>> TEST shape: {x_test.shape}")
    # print(f">>> (before preprocess) NaN in DEV: {np.isnan(x_dev).sum()}")
    # print(f">>> (before preprocess) NaN in TEST: {np.isnan(x_test).sum()}")


    # ==========================================================
    # A. TRAIN ON FULL DEV -> EVALUATE ON TEST
    # ==========================================================
    if pipeline_type == "heldout":
        x_dev, imputer, scaler = preprocess_feature(x_dev)
        x_test = imputer.transform(x_test)
        x_test = scaler.transform(x_test)

        df_result_holdout = pipeline_heldout(
            x_train=x_dev,
            y_train=y_dev,
            x_test=x_test,
            y_test=y_test,
            groups_test=groups_test,
            model_type=model_type
        )

        print(
            "\n[HELD-OUT TEST RESULT]\n", 
            df_result_holdout[["model_name", "score", metric]]
        )
        os.makedirs(path_save, exist_ok=True)
        filename_holdout = os.path.join(path_save, f"{group_feature}_{task}_heldout.csv")
        df_result_holdout.to_csv(filename_holdout, index=False)
        print(f"\n>>> Saved held-out result to: {filename_holdout}")



    # ==========================================================
    # 5-FOLD CV ON FULL DATASET (DEV + TEST)
    # ==========================================================
    elif "kfold" in pipeline_type:

        data_all = pd.concat(
            [data_dev, data_test],
            ignore_index=True
        )

        x_all, y_all, groups_all, _ = dataClass_baseline.get_data_by_task(
            data_all,
            task,
            list_features
        )
    
        print("\n==== FULL DATASET FOR CV ====")
        print("X shape:", x_all.shape)
        print("Y shape:", y_all.shape)
        print("Subjects:", len(np.unique(groups_all)))

        if pipeline_type == "repeated_kfold":
            df_result_cv = pipeline_repeated_kfold(
                x=x_all,
                y=y_all,
                groups=groups_all,
                model_type=model_type,
                n_splits=n_splits,
                n_repeats=n_repeats
            )
            filename_cv = os.path.join(path_save, f"{group_feature}_{task}_{n_splits}foldcv_repeated{n_repeats}.csv")
        else:
            df_result_cv = pipeline_kfold(
                x=x_all,
                y=y_all,
                groups=groups_all,
                model_type=model_type,
                n_splits=n_splits
            )
            filename_cv = os.path.join(path_save, f"{group_feature}_{task}_{n_splits}foldcv.csv")


        ##
        df_result_cv.to_csv(filename_cv, index=False)
        summary = (
            df_result_cv
            .groupby("model_name")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        print(f"\n[DEV CV] [{metric}] Mean ± Std")
        print(summary)
        print(f"\n>>> Saved CV result to: ", f"{filename_cv}")



    else:
        raise ValueError(f"Unsupported pipeline_type: {pipeline_type}")