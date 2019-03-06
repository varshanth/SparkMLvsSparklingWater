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
predictor_columns = ds_f.drop(target_col_name).col_names
response_column = target_col_name



print('----Training Model----')
from h2o.transforms.decomposition import H2OPCA
pca_decomp = H2OPCA(k=10, transform="NONE", pca_method="Power", impute_missing=True)
pca_decomp.train(x=predictor_columns, training_frame=ds_f)
pca_decomp.summary()



print('----Testing Model----')
predictions = pca_decomp.predict(ds_test_f)
print(predictions.ncols)

print('----End----')



