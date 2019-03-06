import argparse

def parse_ds_args(num_train_chunks, num_test_chunks, chunksize=100000):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv', type=str, required=True)
    parser.add_argument('--chunksize', type=int, default=chunksize)
    parser.add_argument('--num_train_chunks', type=int, default=num_train_chunks)
    parser.add_argument('--num_test_chunks', type=int, default=num_test_chunks)
    args = parser.parse_args()
    return args

