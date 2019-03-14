import pandas as pd
from .datasets_conf import datasets_info as ds_info

def csv_to_df(path_to_csv, chunksize, num_train_chunks, num_test_chunks, dataset='susy'):
    pd_df_chunks = pd.read_csv(path_to_csv, header=None, chunksize=chunksize)
    chunk_list = []
    for i, chunk in enumerate( pd_df_chunks ):
        if i == num_train_chunks + num_test_chunks:
            break
        chunk_list.append( chunk )
    train_pd_df = pd.concat( chunk_list[:num_train_chunks] )
    test_pd_df = pd.concat( chunk_list[num_train_chunks:])
    target_col_idx = 0
    if 'target_col_idx' in ds_info[dataset]:
        target_col_idx = ds_info[dataset]['target_col_idx']
    target_col_name = "target"
    feature_col_names = ['{0}'.format(idx) for idx in range(ds_info[dataset]['num_cols'])]
    return train_pd_df, test_pd_df, target_col_name, target_col_idx, feature_col_names


def csv_to_df_chunks(path_to_csv, chunksize, num_train_chunks, num_test_chunks, dataset='susy'):
    pd_df_chunks = pd.read_csv(path_to_csv, header=None, chunksize=chunksize)
    chunk_list = []
    for i, chunk in enumerate( pd_df_chunks ):
        if i == num_train_chunks + num_test_chunks:
            break
        chunk_list.append( chunk )
    target_col_idx = 0
    if 'target_col_idx' in ds_info[dataset]:
        target_col_idx = ds_info[dataset]['target_col_idx']
    target_col_name = "target"
    feature_col_names = ['{0}'.format(idx) for idx in range(ds_info[dataset]['num_cols'])]
    return chunk_list[:num_train_chunks], chunk_list[num_train_chunks:], target_col_name, target_col_idx, feature_col_names
