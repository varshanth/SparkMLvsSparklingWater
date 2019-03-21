#!/usr/bin/env python
# coding: utf-8

from app_argparse import parse_args
from datasets.load_data_into_df import csv_to_df
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
    trainingSummary = lrm.summary
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    roc = trainingSummary.roc.toPandas()
    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    log_with_time('Training set areaUnderROC: ' + f"{trainingSummary.areaUnderROC}")
    '''
    log_with_time('Training Accuracy ' + f"{trainingSummary.accuracy}")
    return lrm


def _test_logistic_regression_model(logistic_regression_model, feat_test):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    predictions = logistic_regression_model.transform(feat_test)
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    pred_and_labels = predictions.select('prediction', 'label')
    log_with_time(f"Test Set Accuracy: {evaluator.evaluate(pred_and_labels)}")
    return None


def _get_mlp_model(feat_train):
    from pyspark.ml.classification import MultilayerPerceptronClassifier
    num_feat = len(feat_train.first()["features"])
    layers = [num_feat, 10, 10, 2]
    mlp_trainer = MultilayerPerceptronClassifier(
            maxIter=10, layers=layers, seed=123, stepSize=0.005, solver='gd',
            featuresCol="features", labelCol="label")
    mlp_model = mlp_trainer.fit(feat_train)
    return mlp_model


def _test_mlp_model(mlp_model, feat_test):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    predictions = mlp_model.transform(feat_test)
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    pred_and_labels = predictions.select('prediction', 'label')
    log_with_time(f"Test Set Accuracy: {evaluator.evaluate(pred_and_labels)}")
    return None


def _get_pca_model(feat_train):
    from pyspark.ml.feature import PCA
    pca = PCA(k=10, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(feat_train)
    return pca_model


def _test_pca_model(pca_model, feat_test):
    pca_model.transform(feat_test).collect()[0].pca_features
    log_with_time('Explained Variance: ' + f"{pca_model.explainedVariance}")
    return None

_model_fn_call_map = {
        'kmeans': {'train': _get_kmeans_model, 'test': _test_kmeans_model},
        'logistic_regression' : {'train': _get_logistic_regression_model, 'test': _test_logistic_regression_model},
        'pca' : {'train': _get_pca_model, 'test': _test_pca_model},
        'mlp' : {'train': _get_mlp_model, 'test': _test_mlp_model},
        'all' : {}
        }


if __name__ == '__main__':
    args = parse_args(list(_model_fn_call_map.keys()))

    log_with_time('----Pyspark Execution----')
    log_with_time(f"----Dataset: {args.dataset} Model:{args.model_type}----")
    log_with_time(f"----Chunksize: {args.chunksize} Train Chunks:{args.num_train_chunks} Test Chunks: {args.num_test_chunks}----")

    log_with_time('----Loading Dataset----')

    ds_train_pd_df, ds_test_pd_df, target_col_name, target_col_idx, feature_col_names = csv_to_df(args.path_to_csv,
            args.chunksize, args.num_train_chunks, args.num_test_chunks)
    col_names = [target_col_name] + feature_col_names

    # Merge Train & Test into a single DF for coding simplicity
    import pandas as pd
    ds_merged_pd_df = pd.concat([ds_train_pd_df, ds_test_pd_df])
    train_frac = 1. * args.num_train_chunks/(args.num_test_chunks+args.num_train_chunks)


    log_with_time('----Creating Spark Context----')
    from pyspark import SparkContext
    from pyspark.sql import SQLContext
    sc = SparkContext(args.master_url, f"PySpark_{args.dataset}_{args.model_type}")
    sc.setLogLevel("ERROR")
    sqlCtx = SQLContext(sc)


    log_with_time('----Creating Spark DataFrame----')
    dist_rdd = sc.parallelize(ds_merged_pd_df.values.tolist())
    del(ds_merged_pd_df)
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

    model_type = args.model_type
    model_type_choices = [model_type]
    if model_type == 'all':
        model_type_choices = list(_model_fn_call_map.keys())
    for model_type in model_type_choices:
        if model_type == 'all':
            continue
        try:
            log_with_time(f"----Training {model_type} Model----")
            model = _model_fn_call_map[model_type]['train'](feat_train)

            log_with_time(f"----Testing {model_type} Model----")
            ret_val = _model_fn_call_map[model_type]['test'](model, feat_test)
        except:
            log_with_time(f"---{model_type} Failed---")
            continue

    log_with_time('----End----')
