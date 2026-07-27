from typing import Optional, Dict, List
import os
import pandas as pd
import numpy as np
# utils
from src.utils import utils_data
from src.utils.logger import get_logger
#
import ray
from functools import reduce
import gc
import warnings
warnings.filterwarnings("ignore")
##
logger = get_logger(__name__)


################## UTILS #################
##########################################
##########################
def reform_dataframe(df: pd.DataFrame):
    """
    Reset multi-Index, back to the dataframe with multiple columns.
    """
    return df.reset_index().assign(
        timestamp=lambda x: pd.to_datetime(x["timestamp"]),
        date=lambda x: x["timestamp"].dt.date.astype(str)
    )

##########################
def calc_stats(
    df: pd.DataFrame, 
    col:str
) -> None:
    """Calculate some stats feature on the column
    Args:
        df: dataframe
        col (str): name of column
    """
    return df.groupby(by=["public_id", "date"])[col] \
            .apply(lambda g: [
                g.max(), g.mean(), g.min(), g.skew(), g.std()
                ]
            ) \
            .reset_index()



##########################
def get_feature_by_day(
    stream: str,
    df: pd.DataFrame,
    verbose: bool = False,
    **kwargs, 
) -> pd.DataFrame:
    """Formulate dataframe of feature of each day.
    
    Args:
        stream:     Stream name
        df:         Dataframe for each day of each subject. 

    """
    ## Formulate feature
    df_feature = pd.DataFrame()
    for j, idx in enumerate(df.index):
        # key index
        df_feature.loc[idx, "key"] = df.loc[idx, "public_id"] \
            + "_" + df.loc[idx, "date"]

        # ft name
        ft = df.loc[idx, 'feature']
        for k,v in enumerate(ft):
            df_feature.loc[idx, f"{stream}#{k}"] = v
    
    ##
    if verbose:
        print(f"\n\n======[VERBOSE] get_feature_by_day() =======")
        print(f">>> stream: {stream} | df_feature.shape: {df_feature.shape} | df_feature.columns: {df_feature.columns.tolist()}")


    ## 
    return df_feature.set_index("key")



##########################
## APPSTATE
def _feature_appstate(df:pd.DataFrame, **kwargs):
    """ 
    Compute the total number of locks/unlocks...during the day [Gashi24]
    Calculate (for each day) the total number of Locks & Unlocks
    """
    col = 'appstate'
    df = reform_dataframe(df)

    # Group each date, and count isLock=1 & isLock=0  
    return df.groupby(by=["public_id", "date"], group_keys=True) \
                .apply(lambda g: [
                    g.loc[g[col]==2, col].count(),
                    g.loc[g[col]==1, col].count(),
                    g.loc[g[col]==0, col].count(),
                    ]
                ).reset_index() \
                    .rename(columns={0 : 'feature'})


##########################
## HRTS
def _feature_hrts(df:pd.DataFrame, **kwargs):
    """ 
    Function to get feature HEART RATE TIME SERIES
    """
    col = 'hrts'
    df = reform_dataframe(df)
    # get stats
    return calc_stats(df.loc[df[col] != 0], col) \
            .rename(columns={col : 'feature'})

## ATS
def _feature_azmts(df:pd.DataFrame, **kwargs):
    """ 
    Function to get feature ACTIVITY ZONE MINUTE TIME SERIES
    """
    col = 'azmts'
    df = reform_dataframe(df)
    # get stats
    return calc_stats(df.loc[df[col] != 0], col) \
            .rename(columns={col : 'feature'})


## ATS
def _feature_common(df:pd.DataFrame, **kwargs):
    """ 
    Function to get feature ATS / ACC / GYRO
    """
    df = reform_dataframe(df)

    ## this ats have multiple columns, so we need to calculate stats for each column
    return (
        df.groupby(["public_id", "date"])
        .apply(
            lambda g: [
                *(g[col].mean() for col in kwargs["channels"]),
                *(g[col].min() for col in kwargs["channels"]),
                *(g[col].max() for col in kwargs["channels"]),
                *(g[col].skew() for col in kwargs["channels"]),
                *(g[col].std() for col in kwargs["channels"]),
            ]
        )
        .reset_index(name="feature")
    )


def _feature_summaries(df:pd.DataFrame, **kwargs):
    """ 
    Function to get feature SLEEP / BREATHINGRATE / HRV / SKINTEMP / SPO2
    Just return the value of the column, since these are already daily summaries.
    """
    df = reform_dataframe(df)
    if len(kwargs["channels"]) > 1:
        return (
            df.groupby(["public_id", "date"])
            .apply(
                lambda g: [
                    *(g[col].values.flatten().tolist() for col in kwargs["channels"])
                ]
            )
            .reset_index(name="feature")
        )
    else:
        return (
            df.groupby(["public_id", "date"])
            .apply(
                lambda g: [
                    g[kwargs["channels"]].values.flatten().tolist()
                ]
            )
            .reset_index(name="feature")
        )


def _feature_dailysurvey(df:pd.DataFrame, **kwargs):
    """ 
    Function to get feature:
        DAILYSURVEY (phq2_score, gad2_score), each score range 0-6 (so max_score=6)
    Note that, this feature has several columns, and each day user can log 2-3 entries 
    We will select only the last entry of the day and normalize it
    """
    df = reform_dataframe(df)
    ## first, select last entry of the day
    df = df.groupby(["public_id", "date"]).last().reset_index()
    
    ## then, we normalized
    df[kwargs["channels"]] = df[kwargs["channels"]].apply(lambda x: x / 6)

    ## finally, we return the feature
    return (
        df.groupby(["public_id", "date"])
        .apply(
            lambda g: [
                *(g[col].values.flatten().tolist() for col in kwargs["channels"])
            ]
        )
        .reset_index(name="feature")
    )

def _feature_moodlog(df:pd.DataFrame, max_score:int=6, **kwargs):
    """ 
    Function to get feature:
        MOODLOG (miserypleasure, sleepinessarousal), each score range -2->2 (so we should normalize to 0-1 by adding 2 and divide to 4)
    Note that, this feature has several columns, and each day user can log 2-3 entries 
    We will select only the last entry of the day and normalize
    """
    df = reform_dataframe(df)
    ## first, select last entry of the day
    df = df.groupby(["public_id", "date"]).last().reset_index()
    
    ## then, we normalized
    df[kwargs["channels"]] = df[kwargs["channels"]].apply(lambda x: (x + 2) / 4)

    ## finally, we return the feature
    return (
        df.groupby(["public_id", "date"])
        .apply(
            lambda g: [
                *(g[col].values.flatten().tolist() for col in kwargs["channels"])
            ]
        )
        .reset_index(name="feature")
    )








############## FUNCTION FEATURE MAPPING #################
########################################################
FUNC_FEATURE = {


    ## Pc (phone-continuous)
    'ACC': _feature_common,
    'GYRO': _feature_common,
    "BATTERY": _feature_common,
    "NETWORK": _feature_common,

    ## Pe (phone-event)
    'APPSTATE': _feature_appstate, 

    ## Wm (wearable-1min)
    'AZMTS': _feature_azmts,
    'ATS': _feature_common,
    'HRTS': _feature_hrts,

    ## Ws (Wearable-daily)
    "BREATHINGRATE": _feature_summaries,
    "HRV": _feature_summaries,
    "SKINTEMP": _feature_summaries,
    "SPO2": _feature_summaries,
    "SLEEP": _feature_summaries,

    ## Sd (selfreport-daily)
    "DAILYSURVEY": _feature_dailysurvey,
    "MOODLOG": _feature_moodlog,

    # ## Sw (selfreport-weekly)
    # "PHQ9": _feature_summaries,
    # "GAD7": _feature_summaries,

}









############## RUN FEATURE EXTRACTION #################
########################################################
def _process(
    stream: str,
    subject: str,
    d_path: Dict,
    verbose: bool = True,
    **kwargs,
) -> Optional[pd.Series]:
    """
    Apply processing to the loaded dataframe 
    """
    try:
        fn, config = utils_data.stream_to_fn(stream, **kwargs)

        # Load 
        # df = _load_data(fn, subject, d_path, **kwargs)
        path_save = os.path.join(kwargs['DIR_PROCESSED'], 'sensor_raw', f'{subject}.pkl')
        dict_sensor_raw = utils_data.load(path_save)
        df = dict_sensor_raw[stream]
        del dict_sensor_raw
        if verbose:
            logger.debug(f"[_process()] Loaded sensor: {stream} (shape: {df.shape})\n {df.head()}\n")

        ## process
        data = FUNC_FEATURE[stream](df, **config)
        if verbose:
            logger.debug(f"\n[_process()] Extract feature: {stream} (shape: {data.shape}\n {data.head()}\n")

        ## extract feature 
        data = get_feature_by_day(stream, data, **config)
        if verbose:
            logger.debug(f"\n[_process()] Extracted features by day: {stream} (shape: {data.shape})")
            print(data.head())

        ## log
        logger.info(f"Done Processed: {stream}")
        return {stream: data}
    except:
        logger.error(f"Not found: {stream}")
        return dict()




############### RUN FEATURE EXTRACTION ########
################################################################
def run_process_feature(
    subject: str, 
    d_path: Dict,
    list_streams: List[str],
    num_cpus: int,
    use_ray: bool = False,
    verbose: bool = False,
    **kwargs,
) -> None:
    """
    ray process for 1 subject
    """
    _dir = os.path.join(kwargs['DIR_PROCESSED'], 'feature_original')
    os.makedirs(_dir, exist_ok=True)
    path_save = os.path.join(_dir, f'{subject}.pkl')

    if not os.path.isfile(path_save):
        if use_ray:
            with utils_data.on_ray(num_cpus=num_cpus):
                func = ray.remote(_process).remote

                jobs = [
                    func(stream, subject, d_path, **kwargs) \
                        for stream in list_streams
                ]
                jobs = ray.get(jobs)
                jobs = reduce(lambda a, b: {**a, **b}, jobs)
                
                # save files
                utils_data.dump(jobs, path_save)
                del jobs
                gc.collect()
        else:
            jobs = [_process(stream, subject, d_path, **kwargs) for stream in list_streams]
            utils_data.dump(jobs, path_save)

            jobs = {}
            for stream in list_streams:
                job = _process(
                    stream=stream,
                    subject=subject,
                    d_path=d_path,
                    **kwargs
                )
                jobs.update(job)
            utils_data.dump(jobs, path_save)
            del jobs
            gc.collect()

        
        logger.info(f"[run_process_feature()] Processed: {subject} at {path_save}")


    data = utils_data.load(path_save)
    print(data)
    gc.collect()
    return data