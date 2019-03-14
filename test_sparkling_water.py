#!/usr/bin/env python
# coding: utf-8

from pysparkling import *
from ds_argparse import parse_ds_args
from datasets.load_data_into_df import csv_to_df
from utils import log_with_time

def _get_logistic_regression_model(predictor_col, response_col, train_f, val_f):
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator
    glm_model = H2OGeneralizedLinearEstimator(family="binomial", alpha=[0.5])
    glm_model.train(x = predictor_col, y = response_col,
            training_frame = train_f, validation_frame = val_f)
    return glm_model


def _test_logistic_regression_model(logistic_regression_model, test_f):
    logistic_regression_model.model_performance(test_f)
    predict_table = logistic_regression_model.predict(test_f)
    log_with_time('---- Prediction Done: Manual Evaluation Start---')
    predictions = predict_table.as_data_frame()["predict"].tolist()
    ground_truth = test_f.as_data_frame()["target"].tolist()
    num_hits = 0
    for gt, pred in zip(ground_truth, predictions):
        num_hits += (gt==pred)
    log_with_time('GLM (Binomial) Accuracy = {0}'.format(1.* num_hits /len(ground_truth)))
    return None


def _get_gbm_model(predictor_col, response_col, train_f, val_f):
    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    gbm_model = H2OGradientBoostingEstimator(ntrees       = 50,
                                             max_depth    = 3,
                                             learn_rate   = 0.1,
                                             distribution = "bernoulli"
                                            )
    gbm_model.train(x = predictor_col, y = response_col,
            training_frame = train_f, validation_frame = val_f)
    return gbm_model


def _test_gbm_model(gbm_model, test_f):
    gbm_model.model_performance(test_f)
    predict_table_gbm = gbm_model.predict(test_f)
    log_with_time('---- Prediction Done: Manual Evaluation Start---')
    predictions = predict_table_gbm.as_data_frame()["predict"].tolist()
    ground_truth = test_f.as_data_frame()["target"].tolist()
    num_hits = 0
    for gt, pred in zip(ground_truth, predictions):
        num_hits += (gt==pred)
    log_with_time('GBM Accuracy = {0}'.format(1.* num_hits/len(ground_truth)))
    return None


def _get_kmeans_model(predictor_col, response_col, train_f, val_f):
    from h2o.estimators.kmeans import H2OKMeansEstimator
    kmeans_model = H2OKMeansEstimator(k=2, max_iterations=1000000)
    kmeans_model.train(x = predictor_col, training_frame = train_f,
            validation_frame = val_f)
    return kmeans_model


def _test_kmeans_model(kmeans_model, test_f):
    predict_table = kmeans_model.predict(test_f)
    predictions = predict_table.as_data_frame()["predict"].tolist()
    ground_truth = test_f.as_data_frame()["target"].tolist()
    num_hits = 0
    for gt, pred in zip(ground_truth, predictions):
        num_hits += (gt==pred)
    log_with_time('KMeans Accuracy = {0}'.format(1.* num_hits/len(ground_truth)))
    return None


def _get_pca_model(predictor_col, response_col, train_f, val_f):
    from h2o.transforms.decomposition import H2OPCA
    pca_decomp = H2OPCA(k=10, transform="NONE", pca_method="Power", impute_missing=True)
    pca_decomp.train(x=predictor_columns, training_frame=train_f)
    pca_decomp.summary()
    return pca_decomp


def _test_pca_model(pca_model, test_f):
    predictions = pca_model.predict(test_f)
    log_with_time(predictions.ncols)
    return None

_model_fn_call_map = {
        'kmeans': {'train': _get_kmeans_model, 'test': _test_kmeans_model},
        'logistic_regression' : {'train': _get_logistic_regression_model, 'test': _test_logistic_regression_model},
        'pca' : {'train': _get_pca_model, 'test': _test_pca_model},
        'gbm' : {'train': _get_gbm_model, 'test': _test_gbm_model}
        }

if __name__ == '__main__':
    args = parse_ds_args( list(_model_fn_call_map.keys()) )

    log_with_time('----Loading Dataset----')

    ds_train_pd_df, ds_test_pd_df, target_col_name, target_col_idx, feature_col_names = csv_to_df(
            args.path_to_csv, args.chunksize, args.num_train_chunks, args.num_test_chunks, args.dataset)
    col_names = [target_col_name]+feature_col_names

    log_with_time("----Creating Spark Context----")
    from pyspark import SparkContext
    sc = SparkContext("local", f"PySpark_{args.dataset}_{args.model_type}")
    sc.setLogLevel("ERROR")

    log_with_time('----Creating H2O Context----')
    hc = H2OContext.getOrCreate(sc)

    log_with_time('----Creating H2O Frame----')
    import h2o
    ds_f = h2o.H2OFrame(ds_train_pd_df, column_names=col_names)
    ds_test_f = h2o.H2OFrame(ds_test_pd_df, column_names=col_names)
    ds_test_f[target_col_name] = ds_test_f[target_col_name].asfactor()

    h2o.cluster().timezone = "Etc/UTC"

    log_with_time('----Assembling Data----')
    ds_f[target_col_name] = ds_f[target_col_name].asfactor()
    ds_f_splits = ds_f.split_frame(ratios=[0.8])
    ds_train_f, ds_val_f = ds_f_splits
    predictor_columns = ds_train_f.drop(target_col_name).col_names
    response_column = target_col_name


    log_with_time('----Training Model----')
    model = _model_fn_call_map[args.model_type]['train'](predictor_columns,
            response_column, ds_train_f, ds_val_f)

    log_with_time('----Testing Model----')
    ret_val = _model_fn_call_map[args.model_type]['test'](model, ds_test_f)

    log_with_time('----End----')
