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

print('----Creating H2O Context----')
hc = H2OContext.getOrCreate(spark)

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
# Create and train GBM model
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# Prepare model based on the given set of parameters
gbm_model = H2OGradientBoostingEstimator(ntrees       = 50,
                                         max_depth    = 3,
                                         learn_rate   = 0.1,
                                         distribution = "bernoulli"
                                        )

glm_model = H2OGeneralizedLinearEstimator(family="binomial", alpha=[0.5])

gbm_model.train(x            = predictor_columns,
            y                = response_column,
            training_frame   = ds_train_f,
            validation_frame = ds_val_f
         )

glm_model.train(x            = predictor_columns,
            y                = response_column,
            training_frame   = ds_train_f,
            validation_frame = ds_val_f
         )


print('----Testing Model----')
gbm_model.model_performance(ds_test_f)
glm_model.model_performance(ds_test_f)


print('----Testing Model v2----')
predict_table_gbm = gbm_model.predict(ds_test_f)
predict_table_glm = glm_model.predict(ds_test_f)


predict_table_gbm_df = predict_table_gbm.as_data_frame()
predictions_gbm = predict_table_gbm_df["predict"].tolist()
predict_table_glm_df = predict_table_glm.as_data_frame()
predictions_glm = predict_table_glm_df["predict"].tolist()
ground_truth = ds_test_pd_df[0]
print('GBM Accuracy = {0}'.format(sum(ground_truth==predictions_gbm)/len(ground_truth)))
print('GLM Accuracy = {0}'.format(sum(ground_truth==predictions_glm)/len(ground_truth)))

print('----End----')
