#!/usr/bin/env python
# coding: utf-8


from ds_argparse import parse_ds_args
from load_susy_into_df import susy_csv_to_df

args = parse_ds_args(num_train_chunks=5, num_test_chunks=3)

print('----Loading Dataset----')
ds_train_pd_df, ds_test_pd_df, target_col_name, target_col_idx, feature_col_names = susy_csv_to_df(
        args.path_to_csv, args.chunksize, args.num_train_chunks, args.num_test_chunks)
col_names = [target_col_name]+feature_col_names
import pandas as pd
ds_merged_pd_df = pd.concat([ds_train_pd_df, ds_test_pd_df])
train_frac = 1. * args.num_train_chunks/(args.num_test_chunks+args.num_train_chunks)


print('----Creating Spark Context----')
from pyspark import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext("local", "SparkLogisticRegression")
sc.setLogLevel("ERROR")
sqlCtx = SQLContext(sc)


print('----Creating Spark DataFrame----')
ds_spark_df = sqlCtx.createDataFrame(ds_train_pd_df, schema=col_names)


print('----Assembling Data----')
from pyspark.ml.feature import VectorAssembler
vecassembler = VectorAssembler(
        inputCols=ds_spark_df.columns[:target_col_idx]+ds_spark_df.columns[target_col_idx+1:],
        outputCol="features")
features_vec = vecassembler.transform(ds_spark_df)
features_vec = features_vec.withColumnRenamed(target_col_name, "label")
features_data = features_vec.select("label", "features")
feat_train, feat_test = features_data.randomSplit([train_frac, 1-train_frac])

print('----Training Model----')
from pyspark.ml.classification import LogisticRegression
lrm = LogisticRegression(labelCol="label", featuresCol="features", maxIter=100).fit(feat_train)


import matplotlib.pyplot as plt
import numpy as np

trainingSummary = lrm.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
print('Training Accuracy' + str(trainingSummary.accuracy))

print('----Testing Model----')
from pyspark.ml.evaluation import BinaryClassificationEvaluator
predictions = lrm.transform(feat_test)
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

print('----End----')
