import argparse
from datasets.datasets_conf import datasets_info as ds_info

def parse_ds_args(model_type_choices, num_train_chunks=10, num_test_chunks=10, chunksize=100000):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=model_type_choices, required=True)
    parser.add_argument('--dataset', type=str, choices=list(ds_info.keys()), required=True)
    parser.add_argument('--path_to_csv', type=str, required=True)
    parser.add_argument('--chunksize', type=int, default=chunksize)
    parser.add_argument('--num_train_chunks', type=int, default=num_train_chunks)
    parser.add_argument('--num_test_chunks', type=int, default=num_test_chunks)
    args = parser.parse_args()
    return args

