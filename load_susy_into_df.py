import pandas as pd

def susy_csv_to_df(path_to_csv, chunksize, num_train_chunks, num_test_chunks):
    susy_pd_df_chunks = pd.read_csv(path_to_csv, header=None, chunksize=chunksize)
    susy_chunk_list = []
    for i, chunk in enumerate(susy_pd_df_chunks):
        if i == num_train_chunks+num_test_chunks:
            break
        susy_chunk_list.append(chunk)
    susy_train_pd_df = pd.concat(susy_chunk_list[:num_train_chunks])
    susy_test_pd_df = pd.concat(susy_chunk_list[num_train_chunks:])
    target_col_idx = 0
    target_col_name = "target"
    feature_col_names = ['{0}'.format(idx) for idx in range(18)]
    return susy_train_pd_df, susy_test_pd_df, target_col_name, target_col_idx, feature_col_names
