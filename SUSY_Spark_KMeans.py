#!/usr/bin/env python
# coding: utf-8


from ds_argparse import parse_ds_args
from load_susy_into_df import susy_csv_to_df

args = parse_ds_args(num_train_chunks=5, num_test_chunks=3)

print('----Loading Dataset----')
ds_train_pd_df, ds_test_pd_df, target_col_name, target_col_idx, feature_col_names = susy_csv_to_df(
        args.path_to_csv, args.chunksize, args.num_train_chunks, args.num_test_chunks)
col_names = [target_col_name]+feature_col_names


print('----Creating Spark Context----')
from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
ds_spark_df = sqlCtx.createDataFrame(ds_train_pd_df, schema=col_names)



print('----Assembling Data----')
from pyspark.ml.feature import VectorAssembler
vecassembler = VectorAssembler(
        inputCols=ds_spark_df.columns[:target_col_idx]+ds_spark_df.columns[target_col_idx+1:],
        outputCol="features")
features_vec = vecassembler.transform(ds_spark_df)

features_vec = features_vec.withColumnRenamed(target_col_name, "label")
features_data = features_vec.select("label", "features")
feat_train, feat_test = features_data.randomSplit([0.8, 0.2])



print('----Training Model----')
from pyspark.ml.clustering import KMeans
kmeans = KMeans(featuresCol="features", k=2, maxIter=1000000)
kmeans_model = kmeans.fit(feat_train)


print('----Testing Model----')
from pyspark.ml.evaluation import ClusteringEvaluator
predictions = kmeans_model.transform(feat_test)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
centers = kmeans_model.clusterCenters()
feat = feat_train.collect()[0].features

print('----End----')
