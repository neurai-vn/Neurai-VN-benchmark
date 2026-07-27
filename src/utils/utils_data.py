from typing import Dict, Tuple
import os
import json
import yaml
import ray
import cloudpickle
import sys
from contextlib import contextmanager



#==========================#
# get_dir_info() return dict { 'P0010': (number of files, file size),...}
#==========================#
def get_dir_info(
    data_dir:str
) -> Dict[str, Tuple[str, float]]:
    """Return dict of subject + len(subfiles) & its size"""

    d_len_size = {}
    for sb in subdirs(data_dir): # list of subjects
        path_sb = os.path.join(data_dir, sb)
        d_len_size[sb] = (
            len([entry.name for entry in os.scandir(path_sb)]), # number of files.
            get_tree_size(path_sb)/1e6, # byte*1e6 = MB
        )
    # sort alphabet name
    return dict(sorted(d_len_size.items()))


#==========================#
# subdirs() trả về list các thư mục con trong một thư mục, bỏ qua các thư mục bắt đầu bằng '.'
#==========================#
def subdirs(path):
    """Yield directory names not starting with '.' under given path.
    https://peps.python.org/pep-0471/
    """
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            yield entry.name



#==========================#
# get_tree_size() trả về tổng kích thước của các file trong path và subdirs.
#==========================#
def get_tree_size(path):
    """Return total size of files in path and subdirs. If
    is_dir() or stat() fails, print an error message to stderr
    and assume zero size (for example, file has been deleted).
    https://peps.python.org/pep-0471/
    """
    total = 0
    for entry in os.scandir(path):
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError as error:
            print('Error calling is_dir():', error, file=sys.stderr)
            continue
        if is_dir:
            total += get_tree_size(entry.path)
        else:
            try:
                total += entry.stat(follow_symlinks=False).st_size
            except OSError as error:
                print('Error calling stat():', error, file=sys.stderr)
    return total



#==========================#
# get_dict_path_neuraivn() quét qua cấu trúc thư mục đặc biệt của neuraivn và trả về một dict có cấu trúc:
#==========================#
def get_dict_path_neuraivn(data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Quét cấu trúc: participant/P0010/smartphone/P0010_accelerometer.csv
    Trả về: { 'P0010': {'accelerometer': '/path/to/P0010_accelerometer.csv', ...} }
    """
    d = {}
    participants_dir = os.path.join(data_dir, 'participant')

    if not os.path.exists(participants_dir):
        raise FileNotFoundError(f"Folder {participants_dir} can not be found!")

    with os.scandir(participants_dir) as subject_entries:
        for subject_entry in subject_entries:
            if subject_entry.is_dir():
                sb = subject_entry.name # P0010
                d[sb] = {}
                
                # Duyệt qua các category con (smartphone, wearable, self_report)
                for cat in ['smartphone', 'wearable', 'self_report']:
                    cat_path = os.path.join(subject_entry.path, cat)
                    if os.path.exists(cat_path):
                        with os.scandir(cat_path) as file_entries:
                            for f in file_entries:
                                if f.is_file() and f.name.endswith(".csv"):
                                    # Lấy tên file (vd: P0010_accelerometer.csv) 
                                    # Loại bỏ subject_id để lấy key là 'accelerometer'
                                    modality_key = f.name.replace(f"{sb}_", "")
                                    d[sb][modality_key] = f.path
    return d



#==========================#
# on_ray() là một context manager để khởi tạo và tắt Ray
#==========================#
@contextmanager
def on_ray(*args, **kwargs):
    try:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(*args, **kwargs)
        yield None
    finally:
        ray.shutdown()




#==========================#
# load_yaml() tải dữ liệu từ file YAML và trả về dict
#==========================#
def load_yaml(filename: str) -> None:
    """Load data from YAML"""

    with open(filename) as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)


#==========================#
# load_json() tải dữ liệu từ file JSON và trả về dict
#==========================#
def load_json(filename: str) -> None:
    """Load data from JSON"""

    with open(filename) as file:
        try:
            data = json.load(file)
            return data
        except json.JSONError as exc:
            raise exc



################
def dump(obj, path: str):
    with open(path, mode='wb') as f:
        cloudpickle.dump(obj, f)


################
def load(path: str):
    with open(path, mode='rb') as f:
        return cloudpickle.load(f)




#==========================#
# stream_to_fn() tìm file name tương ứng với stream name trong dict path
#==========================#
def stream_to_fn(stream:str, **kwargs):
    """find the files according to stream name"""

    flag = False
    for fn in kwargs['FILES'].keys():
        mod = kwargs['FILES'][fn]["modality"]
        if mod is not None and stream in mod.keys():
            flag = True
            return fn, mod[stream]
    
    if not flag:
        raise FileNotFoundError(f"Can not find {stream} !")
    


#==========================#
# get_all_stream_name() trả về list tất cả stream name có trong dict path
#==========================#
def get_all_stream_name(**kwargs):
    """return the list of all stream name"""
    
    streams = []
    for fn in kwargs['FILES'].keys():
        mod = kwargs['FILES'][fn]["modality"]
        if mod is not None:
            streams += list(mod.keys())
    return streams