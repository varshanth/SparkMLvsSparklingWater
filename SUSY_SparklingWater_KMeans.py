#!/usr/bin/env python
# coding: utf-8

from pysparkling import *
from ds_argparse import parse_ds_args
from load_susy_into_df import susy_csv_to_df

args = parse_ds_args(num_train_chunks=10, num_test_chunks=10)
print('----Loading Dataset----')
ds_train_pd_df, ds_test_pd_df, target_col_name, target_col_idx, feature_col_names = susy_csv_to_df(
        args.path_to_csv, args.chunksize, args.num_train_chunks, args.num_test_chunks)
col_names = [target_col_name]+feature_col_names

print("----Creating Spark Context----")
from pyspark import SparkContext
sc = SparkContext("local", "SparklingWaterKmeans")
sc.setLogLevel("ERROR")

print('----Creating H2O Context----')
hc = H2OContext.getOrCreate(sc)

print('----Creating H2O Frame----')
import h2o
ds_f = h2o.H2OFrame(ds_train_pd_df, column_names=col_names)
ds_test_f = h2o.H2OFrame(ds_test_pd_df, column_names=col_names)
ds_test_f[target_col_name] = ds_test_f[target_col_name].asfactor()

h2o.cluster().timezone = "Etc/UTC"

print('----Assembling Data----')
ds_f[target_col_name] = ds_f[target_col_name].asfactor()
ds_f_splits = ds_f.split_frame(ratios=[0.8])
ds_train_f, ds_val_f = ds_f_splits
predictor_columns = ds_train_f.drop(target_col_name).col_names
response_column = target_col_name



print('----Training Model----')
from h2o.estimators.kmeans import H2OKMeansEstimator

kmeans_model = H2OKMeansEstimator(k=2, max_iterations=1000000)

kmeans_model.train(x            = predictor_columns,
            training_frame   = ds_train_f,
            validation_frame = ds_val_f
         )


print('----Testing Model----')
predict_table = kmeans_model.predict(ds_test_f)
predict_table_df = predict_table.as_data_frame()
predictions = predict_table_df["predict"].tolist()
ground_truth = ds_test_pd_df[0]
print('KMeans Accuracy = {0}'.format(sum(ground_truth==predictions)/len(ground_truth)))

print('----End----')

