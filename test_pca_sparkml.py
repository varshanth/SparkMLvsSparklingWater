#!/usr/bin/env python
# coding: utf-8

from app_argparse import parse_args
from datasets.load_data_into_df import csv_to_df
from utils import logger, SparkConfig
from datasets.datasets_conf import datasets_info

def _get_pca_model(feat_train, k):
    from pyspark.ml.feature import PCA
    pca = PCA(k = k, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(feat_train)
    # Explained Variance
    logr.log_event('Training Accuracy', f"{sum(pca_model.explainedVariance)}")
    return pca_model

def _test_pca_model(pca_model, feat_test):
    # pca_model.transform(feat_test).collect()[0].pca_features
    pca_model.transform(feat_test)
    return None

_model_fn_call_map = {
        'pca' : {'train': _get_pca_model, 'test': _test_pca_model}
        }

if __name__ == '__main__':
    args = parse_args(list(_model_fn_call_map.keys()))
    num_features = datasets_info[args.dataset]['num_cols']
    logr = logger(args.json_log_file)

    logr.log_event('Library', 'PySparkML')
    logr.log_event('Dataset', f"{args.dataset}")
    logr.log_event('Model', f"{args.model_type}")
    logr.log_event('Chunksize', f"{args.chunksize}")
    logr.log_event('Num train chunks', f"{args.num_train_chunks}")
    logr.log_event('Num test chunks', f"{args.num_test_chunks}")

    logr.start_timer()
    logr.log_event('Loading Dataset')

    ds_train_pd_df, ds_test_pd_df, target_col_name, target_col_idx, feature_col_names = csv_to_df(args.path_to_csv,
            args.chunksize, args.num_train_chunks, args.num_test_chunks)
    col_names = [target_col_name] + feature_col_names

    # Merge Train & Test into a single DF for coding simplicity
    import pandas as pd
    ds_merged_pd_df = pd.concat([ds_train_pd_df, ds_test_pd_df])
    train_frac = 1. * args.num_train_chunks/(args.num_test_chunks+args.num_train_chunks)

    logr.log_event('Creating Spark Session')
    spark_conf = SparkConfig(args.master_url, args.dataset, args.model_type)
    spark = spark_conf.create_spark_session()

    sc = spark.sparkContext
    spark.sparkContext.setLogLevel("ERROR")
    from pyspark.sql import SQLContext
    sqlCtx = SQLContext(sc)

    logr.log_event('Creating Spark DataFrame')
    dist_rdd = sc.parallelize(ds_merged_pd_df.values.tolist())
    del(ds_merged_pd_df)
    ds_spark_df = sqlCtx.createDataFrame(dist_rdd, schema=col_names)

    logr.log_event('Assembling Data')
    from pyspark.ml.feature import VectorAssembler
    vecassembler = VectorAssembler(
            inputCols=ds_spark_df.columns[:target_col_idx]+ds_spark_df.columns[target_col_idx+1:],
            outputCol="features")
    features_vec = vecassembler.transform(ds_spark_df)
    features_vec = features_vec.withColumnRenamed(target_col_name, "label")
    features_data = features_vec.select("label", "features")
    feat_train, feat_test = features_data.randomSplit([train_frac, 1-train_frac])

    model_type = args.model_type
    model_type_choices = [model_type]

    for k_pca in range(10, 51, 10):
        try:
            logr.log_event(f"Training", f"pca_{k_pca}")
            model = _model_fn_call_map[model_type]['train'](feat_train, k_pca)

            logr.log_event(f"Testing",  f"pca_{k_pca}")
            ret_val = _model_fn_call_map[model_type]['test'](model, feat_test)
            logr.log_event("Result", True)
        except Exception as e:
            print(f"Error Has occured: {e}")
            logr.log_event("Result", False)
            continue

    logr.stop_timer()
