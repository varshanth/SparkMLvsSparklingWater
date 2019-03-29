#!/usr/bin/env python
# coding: utf-8

from pysparkling import *
from app_argparse import parse_args
from datasets.load_data_into_df import csv_to_df
from utils import logger, SparkConfig
from datasets.datasets_conf import datasets_info

logr = None

def _get_pca_model(predictor_col, response_col, train_f, val_f):
    from h2o.transforms.decomposition import H2OPCA
    k = 10
    pca_decomp = H2OPCA(k = k, transform="NONE", pca_method="Power",
            impute_missing=True)
    pca_decomp.train(x=predictor_columns, training_frame=train_f)
    pca_decomp.summary()
    # Explained Variance
    logr.log_event(f'Training Accuracy', f'{pca_decomp.varimp()[2][k-1]}')
    return pca_decomp


def _test_pca_model(pca_model, test_f):
    predictions = pca_model.predict(test_f)
    # logr.log_event(predictions.ncols)
    return None

_model_fn_call_map = {
        'pca' : {'train': _get_pca_model, 'test': _test_pca_model}
        }

if __name__ == '__main__':
    args = parse_args(list(_model_fn_call_map.keys()))
    num_features = datasets_info[args.dataset]['num_cols']
    logr = logger(args.json_log_file)

    logr.log_event('Library', 'PySparkling Water')
    logr.log_event('Dataset', f"{args.dataset}")
    logr.log_event('Model', f"{args.model_type}")
    logr.log_event('Chunksize', f"{args.chunksize}")
    logr.log_event('Num train chunks', f"{args.num_train_chunks}")
    logr.log_event('Num test chunks', f"{args.num_test_chunks}")

    logr.start_timer()
    logr.log_event('Loading Dataset')

    ds_train_pd_df, ds_test_pd_df, target_col_name, target_col_idx, feature_col_names = csv_to_df(
            args.path_to_csv, args.chunksize, args.num_train_chunks, args.num_test_chunks, args.dataset)
    col_names = [target_col_name]+feature_col_names

    logr.log_event("Creating Spark Session")
    spark_conf = SparkConfig(args.master_url, args.dataset, args.model_type)
    spark = spark_conf.create_spark_session()

    spark.sparkContext.setLogLevel("ERROR")
    from pyspark.sql import SQLContext

    logr.log_event('Creating H2O Context')
    hc = H2OContext.getOrCreate(spark)

    logr.log_event('Creating H2O Frame')
    import h2o
    ds_f = h2o.H2OFrame(ds_train_pd_df, column_names=col_names)
    ds_test_f = h2o.H2OFrame(ds_test_pd_df, column_names=col_names)
    ds_test_f[target_col_name] = ds_test_f[target_col_name].asfactor()

    h2o.cluster().timezone = "Etc/UTC"

    logr.log_event('Assembling Data')
    ds_f[target_col_name] = ds_f[target_col_name].asfactor()
    ds_f_splits = ds_f.split_frame(ratios=[0.8])
    ds_train_f, ds_val_f = ds_f_splits
    predictor_columns = ds_train_f.drop(target_col_name).col_names
    response_column = target_col_name

    model_type = args.model_type
    model_type_choices = [model_type]

    for k_pca in range(10, 51, 10):
        try:
            logr.log_event(f"Training", f"pca_{k_pca}")
            model = _model_fn_call_map[model_type]['train'](predictor_columns,
                    response_column, ds_train_f, ds_val_f, k_pca)

            logr.log_event(f"Testing",  f"pca_{k_pca}")
            ret_val = _model_fn_call_map[model_type]['test'](model, ds_test_f)
            logr.log_event("Result", True)
        except Exception as e:
            print(f"Error Has occured: {e}")
            logr.log_event("Result", False)
            continue

    logr.stop_timer()
