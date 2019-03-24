#!/usr/bin/env python
# coding: utf-8

from pysparkling import *
from app_argparse import parse_args
from datasets.load_data_into_df import csv_to_df
from utils import logger

logr = None

def _get_logistic_regression_model(predictor_col, response_col, train_f, val_f):
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator
    glm_model = H2OGeneralizedLinearEstimator(family="binomial", alpha=[0.5],
            max_iterations = 50)
    glm_model.train(x = predictor_col, y = response_col,
            training_frame = train_f, validation_frame = val_f)
    logr.log_event('Training Accuracy', f"{glm_model.accuracy()[0][1]}")
    return glm_model


def _test_logistic_regression_model(logistic_regression_model, test_f):
    logistic_regression_model.model_performance(test_f)
    predict_table = logistic_regression_model.predict(test_f)
    logr.log_event('Manual Evaluation')
    predictions = predict_table.as_data_frame()["predict"].tolist()
    ground_truth = test_f.as_data_frame()["target"].tolist()
    num_hits = 0
    for gt, pred in zip(ground_truth, predictions):
        num_hits += (gt==pred)
    logr.log_event('Testing Accuracy', f"{1.* num_hits/len(ground_truth)}")
    return None


def _get_mlp_model(predictor_col, response_col, train_f, val_f):
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator
    mlp_model = H2ODeepLearningEstimator(activation = 'tanh',
            adaptive_rate=False, nesterov_accelerated_gradient=False,
            hidden = [10,10], seed = 123, epochs = 10)
    mlp_model.train(x = predictor_col, y = response_col,
            training_frame = train_f, validation_frame = val_f)
    return mlp_model


def _test_mlp_model(mlp_model, test_f):
    mlp_model.model_performance(test_f)
    predict_table_mlp = mlp_model.predict(test_f)
    logr.log_event('Manual Evaluation')
    predictions = predict_table_mlp.as_data_frame()["predict"].tolist()
    ground_truth = test_f.as_data_frame()["target"].tolist()
    num_hits = 0
    for gt, pred in zip(ground_truth, predictions):
        num_hits += (gt==pred)
    logr.log_event('Testing Accuracy', f"{1.* num_hits/len(ground_truth)}")
    return None


def _get_kmeans_model(predictor_col, response_col, train_f, val_f):
    from h2o.estimators.kmeans import H2OKMeansEstimator
    kmeans_model = H2OKMeansEstimator(k=2, max_iterations=1000000)
    kmeans_model.train(x = predictor_col, training_frame = train_f,
            validation_frame = val_f)
    return kmeans_model


def _test_kmeans_model(kmeans_model, test_f):
    predict_table = kmeans_model.predict(test_f)
    '''
    predictions = predict_table.as_data_frame()["predict"].tolist()
    ground_truth = test_f.as_data_frame()["target"].tolist()
    num_hits = 0
    for gt, pred in zip(ground_truth, predictions):
        num_hits += (gt==pred)
    logr.log_event('KMeans Accuracy = {0}'.format(1.* num_hits/len(ground_truth)))
    '''
    return None


def _get_pca_model(predictor_col, response_col, train_f, val_f):
    from h2o.transforms.decomposition import H2OPCA
    pca_decomp = H2OPCA(k=10, transform="NONE", pca_method="Power",
            impute_missing=True)
    pca_decomp.train(x=predictor_columns, training_frame=train_f)
    pca_decomp.summary()
    return pca_decomp


def _test_pca_model(pca_model, test_f):
    predictions = pca_model.predict(test_f)
    # logr.log_event(predictions.ncols)
    return None

_model_fn_call_map = {
        'kmeans': {'train': _get_kmeans_model, 'test': _test_kmeans_model},
        'logistic_regression' : {
            'train': _get_logistic_regression_model,
            'test': _test_logistic_regression_model
            },
        'pca' : {'train': _get_pca_model, 'test': _test_pca_model},
        'mlp' : {'train': _get_mlp_model, 'test': _test_mlp_model},
        'all' : {}
        }

if __name__ == '__main__':
    args = parse_args(list(_model_fn_call_map.keys()))
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

    logr.log_event("Creating Spark Context")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
            .master(args.master_url) \
            .appName( f"PySpark_{args.dataset}_{args.model_type}") \
            .getOrCreate()
    '''
            .config("spark.memory.offHeap.enabled", True) \
            .config("spark.memory.offHeap.size","16g") \
            .config("spark.cleaner.periodicGC.interval", "1min") \
            .getOrCreate()
    '''
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
    if model_type == 'all':
        model_type_choices = list(_model_fn_call_map.keys())
    for model_type in model_type_choices:
        if model_type == 'all':
            continue
        try:
            logr.log_event(f"Training", f"{model_type}")
            model = _model_fn_call_map[model_type]['train'](predictor_columns,
                    response_column, ds_train_f, ds_val_f)

            logr.log_event(f"Testing",  f"{model_type}")
            ret_val = _model_fn_call_map[model_type]['test'](model, ds_test_f)
            logr.log_event("Result", True)
        except Exception as e:
            print(f"Error Has occured: {e}")
            logr.log_event("Result", False)
            continue

    logr.stop_timer()
