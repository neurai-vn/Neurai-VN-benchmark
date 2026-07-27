import pandas as pd
import os
from src.utils import utils_data


##########################
def get_user_info(**kwargs) -> None:
    """get user information from the original demographic file."""

    # check folder
    d_len_size = utils_data.get_dir_info(kwargs['DIR_RAW'])
    print("\n>>> Check folder:\n", d_len_size)
    




################################
def process_label(**kwargs) -> None:
    """
    Extract label
    """
    ui = USER_INFO(**kwargs)
    ui.flow()




################################
class USER_INFO:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        # read study info 
        self.df_info = pd.read_csv(
            os.path.join(kwargs['DIR_RAW'], "metadata/study_info.csv")
        )
        # read annotation
        self.df_annotation = pd.read_csv(
            os.path.join(kwargs['DIR_RAW'], "metadata/annotation.csv")
        )

        # get the demographic sheet 
        print("\n>>> df study_info:\n", self.df_info)
        # get the demographic sheet 
        print("\n>>> df annotation:\n", self.df_annotation)


    def flow(self) -> None:
        # merge
        df = self.df_info.merge(
            self.df_annotation, 
            on='public_id', 
            how='left'
        )
        print("\n>>> df merged:\n", df)

        # save clean csv
        path_save = os.path.join(self.kwargs['DIR_PROCESSED'], 'label_clean.csv')
        df.to_csv(path_save, index=False)
        print(f"\n>>> Save label information to: {path_save}")
