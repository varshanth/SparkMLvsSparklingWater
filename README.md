# SparkMLvsSparklingWater
Performance comparison scripts for SparkML vs SparklingWater

## How to run the .py files

### PySpark
Run from anywhere:  
  

spark-submit \<PY File\> --path_to_csv \<Path to Dataset CSV File\> --chunksize \<Chunksize\> --num_train_chunks=\<Number of Training Chunks\> --num_test_chunks=\<Number of Testing Chunks\>

### PySparkling Water
Run from SparklingWater Home:  
  
  
bin/run-python-script.sh  \<PY File\> --path_to_csv \<Path to Dataset CSV File\> --chunksize \<Chunksize\> --num_train_chunks=\<Number of Training Chunks\> --num_test_chunks=\<Number of Testing Chunks\>

