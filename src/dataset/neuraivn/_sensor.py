from typing import Optional, Dict, List
import os
import pandas as pd
import numpy as np
# utils
from src.utils import utils_data
from src.utils.logger import get_logger
##
from functools import reduce
import ray
import gc
import warnings
warnings.filterwarnings("ignore")
logger = get_logger(__name__)


#===========================#
# Function 
#===========================#
def _proc_appstate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý đặc thù cho stream APPSTATE:
    - Chuyển đổi giá trị 'appstate' thành one-hot encoding (inactive, background, active)
    - Giữ lại timestamp và identifier
    """
    # 'appstate' values (0, 1, 2) for background, inactive, active respectively.
    df_processed = df.copy()
    
    # Map appstate to numeric values
    state_mapping = {
        'background': 0,
        'inactive': 1,
        'active': 2,
    }
    ## if appstate is not None
    df_processed['appstate'] = df_processed['appstate'].map(state_mapping)

    # sort timestamp
    df_processed = df_processed.sort_index()
    
    return df_processed


#===========================#
# Function 
#===========================#
def _proc_network(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý đặc thù cho stream NETWORK:
    - Chuyển đổi giá trị 'network' thành one-hot encoding (WIFI, CELLULAR)
    - Giữ lại timestamp và identifier
    """
    df_processed = df.copy()
    
    ## Map network to numeric values
    network_mapping = {
        'WIFI': 1,
        'CELLULAR': 2,
    }
    df_processed['type'] = df_processed['type'].map(network_mapping)

    ## Map boolen true false in other columns
    bool_cols = ['isConnected', 'isInternetReachable']
    for col in bool_cols:
        df_processed[col] = df_processed[col].astype(int)  # True -> 1, False -> 0


    # sort timestamp
    df_processed = df_processed.sort_index()
    
    return df_processed






#===========================#
# Function 
#===========================#
FUNC_PROC = {
    'APPSTATE': _proc_appstate,
    'NETWORK': _proc_network,
}


#===========================#
# Function 
#===========================#
def _load_data(
    fn: str,
    subject:str,
    d_path:Dict,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """Load data for 1 subject and 1 stream"""

    path_csv = d_path[subject][fn]
    identifier = kwargs['IDENTIFIER']
    info = kwargs['FILES'][fn]['info']

    df = pd.read_csv(path_csv) \
            .assign(**{identifier: subject}) \
            .assign(
                timestamp=lambda x: pd.to_datetime(
                    x[info['ts_col']], 
                    # unit='s',  <-- XÓA DÒNG NÀY ĐI
                    utc=True,
                    errors='coerce',
                ).dt.tz_convert(info['timezone'])
            ) \
            .set_index([identifier, 'timestamp'])
    return df




#===========================#
# Function 
#===========================#
def _process(
    stream: str,
    subject: str,
    d_path: Dict,
    **kwargs,
) -> Dict:
    """
    Xử lý modality stream với logic batch-aware.
    """
    try:
        # 1. Load data (Sử dụng cấu trúc subject/category/file)
        # Tại đây, logic stream_to_fn cần map 'PHY_accel' -> 'PXXXX_accelerometer.csv'
        fn, config = utils_data.stream_to_fn(stream, **kwargs)
        print(f"\nLoading stream: {stream} for subject: {subject} using fn: {fn}")
        print(f"\nConfig for stream {stream}:", config)
        
        # Load (đảm bảo hàm này tìm file theo path mới của neuraivn)
        df = _load_data(fn, subject, d_path, **kwargs)
        # print(f"\nLoaded data for stream {stream} (shape: {df.shape}):\n", df.head())

        # 2. Xử lý Batch-based cleaning
        # - Nếu có hàm xử lý đặc thù cho stream, áp dụng nó
        if stream in FUNC_PROC:
            print(f"Applying specific processing for stream: {stream}")
            df = FUNC_PROC[stream](df)
            # print(f"Data after specific processing for {stream} (shape: {df.shape}):\n", df.head())

        ## remove rows if the date is not in available_dates
        available_dates = utils_data.get_available_dates(subject, d_path, **kwargs)
        mask = np.isin(
            df.index.get_level_values('timestamp').date,
            available_dates
        )
        df = df[mask]
        # print(f"Data after filtering by available dates for {stream} (shape: {df.shape}):\n", df.head())

        # Loại bỏ duplicate timestamp do batch overlap (nếu có)
        df = df[~df.index.duplicated(keep='first')]
        
        # 3. Ép kiểu (Value Measure μ)
        data = df[config['channels']].astype('float32')
        
        logger.info(f"Processed stream: {stream}")
        return {stream: data}

    except Exception as e:
        logger.error(f"Error processing {stream} for {subject}: {e}")
        return {}





# ##########################
def run_process_sensor(
    subject: str, 
    d_path: Dict,
    list_streams: List[str],
    num_cpus: int,
    use_ray: bool = True,
    **kwargs,
) -> Dict:
    """
    
    """
    _dir = os.path.join(kwargs['DIR_PROCESSED'], 'sensor_raw')
    os.makedirs(_dir, exist_ok=True)
    path_save = os.path.join(_dir, f'{subject}.pkl')
    logger.info(f"\n>>> PROCESS RAW SENSOR (Subject: {subject})")

    # get all streams
    utils_data.get_all_stream_name(**kwargs)

    if not os.path.isfile(path_save):

        if use_ray:     

            with utils_data.on_ray(num_cpus=num_cpus):
                func = ray.remote(_process).remote

                jobs = []
                for stream in list_streams:
                    job = func(stream, subject, d_path, **kwargs)
                    jobs.append(job)
                
                jobs = ray.get(jobs)
                ## Merge dictionary các streams
                # data = {k: v for d in jobs for k, v in d.items()}
                jobs = reduce(lambda a, b: {**a, **b}, jobs)
                ## save file
                utils_data.dump(jobs, path_save)
                del jobs
                gc.collect()

        else:
            jobs = {}
            for stream in list_streams:
                job = _process(
                    stream=stream,
                    subject=subject,
                    d_path=d_path,
                    **kwargs
                )
                print(job)
                jobs.update(job)
            utils_data.dump(jobs, path_save)
            del jobs
            gc.collect()


    ### load data anyway
    data = utils_data.load(path_save)
    print(data)
    # del data
    gc.collect()
    return data