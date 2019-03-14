'''
SUSY dataset can be found at UCI repository https://archive.ics.uci.edu/ml/datasets/SUSY.
CATS_DOGS dataset has been created from the python file 'create_cats_dogs_dataset'.
'''
import pandas as pd

def csv_to_df(path_to_csv, chunksize, num_train_chunks, num_test_chunks, dataset='susy'):
    pd_df_chunks = pd.read_csv(path_to_csv, header=None, chunksize=chunksize)
    chunk_list = []
    for i, chunk in enumerate( pd_df_chunks ):
        if i == num_train_chunks + num_test_chunks:
            break
        chunk_list.append( chunk )
    train_pd_df = pd.concat( chunk_list[:num_train_chunks] )
    test_pd_df = pd.concat( chunk_list[num_train_chunks:] )
    target_col_idx = 0

    if dataset == 'susy':
        target_col_name = "target"
        feature_col_names = ['{0}'.format(idx) for idx in range(18)]
    elif dataset == 'cats_dogs':
        target_col_name = "label"
        feature_col_names = ['{0}'.format(idx) for idx in range(25088)]
    return train_pd_df, test_pd_df, target_col_name, target_col_idx, feature_col_names


def csv_to_df_chunks(path_to_csv, chunksize, num_train_chunks, num_test_chunks, dataset='susy'):
    pd_df_chunks = pd.read_csv(path_to_csv, header=None, chunksize=chunksize)
    chunk_list = []
    for i, chunk in enumerate( pd_df_chunks ):
        if i == num_train_chunks + num_test_chunks:
            break
        chunk_list.append( chunk )
    target_col_idx = 0

    if dataset == 'susy':
        target_col_name = "target"
        feature_col_names = ['{0}'.format(idx) for idx in range(18)]
    elif dataset == 'cats_dogs':
        target_col_name = "label"
        feature_col_names = ['{0}'.format(idx) for idx in range(25088)]

    return chunk_list[:num_train_chunks], chunk_list[num_train_chunks:], target_col_name, target_col_idx, feature_col_names
