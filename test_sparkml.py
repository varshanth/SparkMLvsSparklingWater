#!/usr/bin/env python
# coding: utf-8


from ds_argparse import parse_ds_args
from load_susy_into_df import susy_csv_to_df
from utils import log_with_time


def _get_kmeans_model(feat_train):
    from pyspark.ml.clustering import KMeans
    kmeans = KMeans(featuresCol="features", k=2, maxIter=1000000)
    kmeans_model = kmeans.fit(feat_train)
    return kmeans_model


def _test_kmeans_model(kmeans_model, feat_test):
    from pyspark.ml.evaluation import ClusteringEvaluator
    predictions = kmeans_model.transform(feat_test)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    log_with_time("Silhouette with squared euclidean distance = " + f"{silhouette}")
    centers = kmeans_model.clusterCenters()
    return centers


def _get_logistic_regression_model(feat_train):
    from pyspark.ml.classification import LogisticRegression
    lrm = LogisticRegression(labelCol="label", featuresCol="features", maxIter=100).fit(feat_train)
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    trainingSummary = lrm.summary
    roc = trainingSummary.roc.toPandas()
    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    '''
    log_with_time('Training set areaUnderROC: ' + f"{trainingSummary.areaUnderROC}")
    log_with_time('Training Accuracy ' + f"{trainingSummary.accuracy}")
    return lrm


def _test_logistic_regression_model(logistic_regression_model, feat_test):
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    predictions = logistic_regression_model.transform(feat_test)
    evaluator = BinaryClassificationEvaluator()
    log_with_time('Test Area Under ROC' + f"{evaluator.evaluate(predictions)}")
    return None


def _get_pca_model(feat_train):
    from pyspark.ml.feature import PCA
    pca = PCA(k=10, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(feat_train)
    return pca_model


def _test_pca_model(pca_model, feat_test):
    pca_model.transform(feat_test).collect()[0].pca_features
    log_with_time('Explained Variance: '+f"{pca_model.explainedVariance}")
    return None


_dataset_load_map = {
        'susy' : susy_csv_to_df
        }


_model_fn_call_map = {
        'kmeans': {'train': _get_kmeans_model, 'test': _test_kmeans_model},
        'logistic_regression' : {'train': _get_logistic_regression_model, 'test': _test_logistic_regression_model},
        'pca' : {'train': _get_pca_model, 'test': _test_pca_model}
        }


if __name__ == '__main__':
    args = parse_ds_args(list(_dataset_load_map.keys()),
            list(_model_fn_call_map.keys()), num_train_chunks=5, num_test_chunks=3)

    log_with_time('----Loading Dataset----')
    ds_train_pd_df, ds_test_pd_df, target_col_name, target_col_idx, feature_col_names = _dataset_load_map[args.dataset](
            args.path_to_csv, args.chunksize, args.num_train_chunks, args.num_test_chunks)
    col_names = [target_col_name]+feature_col_names

    # Merge Train & Test into a single DF for coding simplicity
    import pandas as pd
    ds_merged_pd_df = pd.concat([ds_train_pd_df, ds_test_pd_df])
    train_frac = 1. * args.num_train_chunks/(args.num_test_chunks+args.num_train_chunks)


    log_with_time('----Creating Spark Context----')
    from pyspark import SparkContext
    from pyspark.sql import SQLContext
    sc = SparkContext("local", f"PySpark_{args.dataset}_{args.model_type}")
    sc.setLogLevel("ERROR")
    sqlCtx = SQLContext(sc)


    log_with_time('----Creating Spark DataFrame----')
    dist_rdd = sc.parallelize(ds_merged_pd_df.values.tolist())
    ds_spark_df = sqlCtx.createDataFrame(dist_rdd, schema=col_names)


    log_with_time('----Assembling Data----')
    from pyspark.ml.feature import VectorAssembler
    vecassembler = VectorAssembler(
            inputCols=ds_spark_df.columns[:target_col_idx]+ds_spark_df.columns[target_col_idx+1:],
            outputCol="features")
    features_vec = vecassembler.transform(ds_spark_df)
    features_vec = features_vec.withColumnRenamed(target_col_name, "label")
    features_data = features_vec.select("label", "features")
    feat_train, feat_test = features_data.randomSplit([train_frac, 1-train_frac])

    log_with_time('----Training Model----')
    model = _model_fn_call_map[args.model_type]['train'](feat_train)

    log_with_time('----Testing Model----')
    ret_val = _model_fn_call_map[args.model_type]['test'](model, feat_test)

    log_with_time('----End----')
