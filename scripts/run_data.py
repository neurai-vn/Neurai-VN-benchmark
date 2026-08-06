"""
Main Script

#######################
Authors: Cuong Pham
Email: cuongquocpham151@gmail.com
"""
from typing import Optional
import os
import pandas as pd
import pytz
import argparse
# custom
from src.utils import utils_data
# neuraivn
import src.dataset.neuraivn._label as f_neuraivn_label
import src.dataset.neuraivn._sensor as f_neuraivn_sensor
import src.dataset.neuraivn._feature as f_neuraivn_feature
import src.benchmarkML.pipeline as f_neuraivn_benchmark
# others
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


##########################
parser = argparse.ArgumentParser()
parser.add_argument('--db', type=str, default='neuraivn', 
    help='name of dataset, e.g., "neuraivn"',
)
parser.add_argument('--mode', type=str, default='info', 
    help='running mode, either "check" or "process"',
)
parser.add_argument('--key', type=str, default=None, 
    help='<sb>-<modality>, e.g., "P01-Acceleration"'
)
parser.add_argument('--task', type=str, default=None, 
    help='task name, e.g., "healthy_pwms"'
)
parser.add_argument('--pipeline_type', type=str, default=None, 
    help='pipeline type, either "heldout" or "kfold"'
)
parser.add_argument('--group_feature', type=str, default=None, 
    help='group feature, e.g., "Wm" or "Wm+Ws"'
)
parser.add_argument('--use_ray', type=str, default=None, 
    help='use ray for parallel processing, true or false"'
)
parser.add_argument('--verbose', type=str, default=None, 
    help='verbose level, true or false"'
)

args = parser.parse_args()



##########################
CONFIG  = {
    'neuraivn': utils_data.load_json('src/configs/neuraivn.json'),

}
FUNC = {
    'neuraivn_label': f_neuraivn_label,
    'neuraivn_sensor': f_neuraivn_sensor,
    'neuraivn_feature-original': f_neuraivn_feature,
    'neuraivn_benchmarkML-baseline': f_neuraivn_benchmark,
}


##########################
def sanity_check(args):
    # check database
    list_db = CONFIG.keys()
    assert args.db in list_db, \
        AssertionError(f'<args.db> must within: {list_db}')




##########################
def main(args):
    """main script"""

    sanity_check(args)
    f_name = f'{args.db}_{args.mode}'
    CF = CONFIG[args.db]
    list_streams =  CF["LIST_STREAMS"]
    use_ray = args.use_ray.lower() == 'true' if args.use_ray is not None else False
    verbose = args.verbose.lower() == 'true' if args.verbose is not None else False

    ## get dict path of all csv files
    d_path = utils_data.get_dict_path_neuraivn(
        data_dir = CF['DIR_RAW']
    )

    ## get list of all data types
    list_files = list(CF['FILES'].keys())

    ## path label information
    path_csv = os.path.join(
        CF['DIR_PROCESSED'], 'label_clean.csv'
    )
    if os.path.isfile(path_csv):
        df = pd.read_csv(path_csv) 
        list_subjects = df[CF['IDENTIFIER']].unique()
    else:
        ## get list of inital subjects
        list_subjects = list(d_path.keys())
    
    ## list streams
    
    # logs
    print(f"\n>>> d_path keys:\n {d_path.keys()}")
    print(f"\n>>> d_path example:\n {next(iter(d_path.values()))}")
    print(f"\n>>> list subjects:\n {list_subjects} (len={len(list_subjects)})")
    print(f"\n>>> AVAILABLE LIST FILES : {list_files}")
    print(f"\n>>> SELECTED LIST STREAMS : {list_streams}")


    ##===============================##
    ## Visually check the raw dataframe of a sensor
    if args.mode == 'checkfile':

        # check args.key
        assert '-' in args.key, \
            "args.key should be <subject>-<fn> format"        
        sb, fn = args.key.split('-')

        # check sb
        assert sb in list_subjects, \
            f"<subject> should be within: {list_subjects}"

        # all types
        if fn == "all":
            for d in list_files:
                df = f_neuraivn_sensor._load_data(
                    fn, sb, d_path, **CF
                )
        # specific type
        elif fn in list_files:
            df = f_neuraivn_sensor._load_data(
                fn, sb, d_path, **CF
            )
            print(f"\n>>>Loaded data {sb} - {fn}")
            print(df)
            print(f"df.shape: ", df.shape)
        # error
        else:
            raise AssertionError(
                "<data_type> must within: ", list_files
            )
    

    ##===============================##
    ## Process sensors for each subject
    elif args.mode == 'label':

        # user info
        df = FUNC[f_name].get_user_info(**CF)
        # label
        FUNC[f_name].process_label(**CF)

    ##===============================##
    elif args.mode == "split":

        identifier = CF["IDENTIFIER"]

        # =====================================================
        # Read label
        # =====================================================
        df_label = pd.read_csv(path_csv)


        # =====================================================
        # Subject source for splitting
        # =====================================================
        df_subject = (
            df_label[[identifier]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # =====================================================
        # Split subjects
        # =====================================================
        dev_subjects, test_subjects = (
            utils_data.split_subjects(
                df=df_subject,
                identifier=identifier,
                test_size=CF["TEST_SIZE"],
                seed=CF["SEED"],
            )
        )
        print(
            f"\n>>> Subjects:"
            f"\n    DEV  = {len(dev_subjects)}"
            f"\n    TEST = {len(test_subjects)}"
        )


        # =====================================================
        # Build label dev/test
        # =====================================================
        df_label_dev = utils_data.filter_by_subjects(
            df_label,
            identifier,
            dev_subjects,
        )

        df_label_test = utils_data.filter_by_subjects(
            df_label,
            identifier,
            test_subjects,
        )

        print(f"\n>>> df_label_dev shape  = {df_label_dev.shape}")
        print(f">>> df_label_test shape = {df_label_test.shape}")

        # =====================================================
        # Save label files
        # =====================================================
        path_save_dev = os.path.join(CF["DIR_PROCESSED"], "label_clean_dev.csv")
        path_save_test = os.path.join(CF["DIR_PROCESSED"], "label_clean_test.csv")
        df_label_dev.to_csv(path_save_dev, index=False)
        df_label_test.to_csv(path_save_test, index=False)
        print(f">>> Saved: {path_save_dev}")
        print(f">>> Saved: {path_save_test}")



    ##===============================##
    ## Process univariate data streams
    elif args.mode == 'sensor':

        # process all subject
        if args.key == 'all':

            for subject in list_subjects:
                FUNC[f_name].run_process_sensor(
                    subject=subject, 
                    d_path=d_path,
                    list_streams=list_streams,
                    num_cpus=16,
                    use_ray=use_ray,
                    **CF,
                )
        # process individual
        elif args.key in list_subjects:
            FUNC[f_name].run_process_sensor(
                subject=args.key, 
                d_path=d_path,
                list_streams=list_streams,
                num_cpus=16,
                use_ray=use_ray,
                **CF,
            )
        else:
            raise KeyError(
                "<key> must within: ", list_subjects
            )
    

    ##===============================##
    elif args.mode == 'feature-original':
        
        # process all subject
        if args.key == 'all':
            for subject in list_subjects:
                FUNC[f_name].run_process_feature(
                    subject=subject, 
                    d_path=d_path,
                    list_streams=list_streams,
                    num_cpus=16,
                    use_ray=use_ray,
                    **CF,
                )
        # process individual
        elif args.key in list_subjects:
            FUNC[f_name].run_process_feature(
                subject=args.key, 
                d_path=d_path,
                list_streams=list_streams,
                num_cpus=16,
                use_ray=use_ray,
                **CF,
            )
        else:
            raise KeyError(
                "<key> must within: ", list_subjects
            )

    ##===============================##
    ## benchmarkML - baseline
    elif args.mode == "benchmarkML-baseline":

        ## runML
        FUNC[f_name].run_ml(
            task = args.task,
            list_streams=list_streams,
            path_save = CF['DIR_RESULTS_ML_BASELINE'],
            pipeline_type = args.pipeline_type, # kfold, repeated_kfold
            group_feature = args.group_feature,
            n_splits = 5,
            n_repeats = 10,
            **CF
        )


    ##===============================##
    else:
        raise NotImplementedError(f'{args.mode} not implemented')







##########################
##########################
if __name__ == "__main__":
    main(args)

