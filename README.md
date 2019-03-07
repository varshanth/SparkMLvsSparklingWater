# SparkMLvsSparklingWater
Performance comparison scripts for SparkML vs SparklingWater

## How to run the .py files

### PySpark

spark-submit \<PY File\> --model_type=\<Type of Model\> --dataset \<Dataset\> --path_to_csv \<Path to Dataset CSV File\> --chunksize \<Chunksize\> --num_train_chunks=\<Number of Training Chunks\> --num_test_chunks=\<Number of Testing Chunks\>

### PySparkling Water
  
$SPARKLING_HOME/bin/run-python-script.sh  \<PY File\> --model_type=\<Type of Model\> --dataset \<Dataset\> --path_to_csv \<Path to Dataset CSV File\> --chunksize \<Chunksize\> --num_train_chunks=\<Number of Training Chunks\> --num_test_chunks=\<Number of Testing Chunks\>


## How to run the Jupyter Notebook Files

PySpark: PYSPARK_DRIVER_PYTHON="ipython" PYSPARK_DRIVER_PYTHON_OPTS="notebook"  pyspark &  
PySparkling Water: PYSPARK_DRIVER_PYTHON="ipython" PYSPARK_DRIVER_PYTHON_OPTS="notebook"  $SPARKLING_HOME/bin/pysparkling &


## Notes
- Python Notebook Files are now deprecated and will not be updated
