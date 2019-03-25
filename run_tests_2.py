import subprocess, time, os
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_slaves', type=str, required=True)
args = parser.parse_args()
num_slaves = args.num_slaves

prod_env = {'SPARK_HOME': '/home/s6singla/spark-2.4.0-bin-hadoop2.7',
            'JAVA_HOME': '/usr/lib/jvm/java-8-oracle',
            'CONDA_PYTHON_EXE': '/home/s6singla/anaconda3/bin/python',
            'SPARKLING_HOME': '/home/s6singla/sparkling-water',
            'PATH': '/home/s6singla/anaconda3/bin:$PATH'}

pyspark_run_cmd = "/home/s6singla/spark-2.4.0-bin-hadoop2.7/bin/spark-submit test_sparkml.py "
pysparkling_run_cmd = "/home/s6singla/sparkling-water/bin/run-python-script.sh test_sparkling_water.py "

model_all_cmd = " --model_type all "

output_dir = "/home/s6singla/SparkMLvsSparklingWater/experiments/"
json_sparkml_file = "sparkml.json"
json_sparkling_file = "sparkling_water.json"

master_url_cmd = "--master_url spark://129.97.173.68:7077"

chunks = {'susy': [('180', '36'), ('360', '72')],
          'cats_dogs_small': [('360', '72'), ('720', '144'), ('1080', '216'), ('1440', '288'), ('1800', '360'), ('2160', '432')],
          'cats_dogs': [('340', '68'), ('680', '136'), ('1020', '204')]
            }
datasets = ['susy', 'cats_dogs_small', 'cats_dogs']

dataset_env = {}
dataset_env['susy'] = {}
dataset_env['susy']['path'] = '/home/s6singla/datasets/SUSY.csv'
dataset_env['susy']['chunksize'] = '10000'
dataset_env['cats_dogs'] = {}
dataset_env['cats_dogs']['path'] = '/home/s6singla/datasets/cats_dogs.csv'
dataset_env['cats_dogs']['chunksize'] = '100'
dataset_env['cats_dogs_small'] = {}
dataset_env['cats_dogs_small']['path'] = '/home/s6singla/datasets/cats_dogs_small.csv'
dataset_env['cats_dogs_small']['chunksize'] = '100'

for dataset in datasets:
    for chunk in chunks[dataset]:
        train_chunks, test_chunks = chunk

        sub_dir = dataset + "_" + train_chunks + "_" + test_chunks + "_" + num_slaves
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            os.chmod(output_dir, 0o777)

        abs_path_sub_dir = output_dir + sub_dir
        if not os.path.exists(abs_path_sub_dir):
            os.mkdir(abs_path_sub_dir)
            os.chmod(abs_path_sub_dir, 0o777)

        abs_path_sparkml_json = abs_path_sub_dir + "/" + json_sparkml_file
        abs_path_sparkling_water_json = abs_path_sub_dir + "/" + json_sparkling_file

        cmd = model_all_cmd + " --path_to_csv=" + dataset_env[dataset]['path'] + " --dataset=" + dataset + " --num_train_chunks=" + train_chunks + \
              " --num_test_chunks=" + test_chunks + " --chunksize=" + dataset_env[dataset]['chunksize'] + " " + master_url_cmd

        pyspark_cmd = pyspark_run_cmd + cmd + " --json_log_file=" + abs_path_sparkml_json
        pysparkling_cmd = pysparkling_run_cmd + cmd + " --json_log_file=" + abs_path_sparkling_water_json

        try:
            print( "Running cmd", pyspark_cmd )
            subprocess.run(pyspark_cmd, shell=True, check=True, env=prod_env)
        except:
            print("Pyspark Failed for chunk ", chunk)

        time.sleep(5)

        try:
            print( "Running cmd", pysparkling_cmd )
            subprocess.run(pysparkling_cmd, shell=True, check=True, env=prod_env)
        except:
            print("Pysparkling Failed for chunk ", chunk)

        time.sleep(5)

        try:
            print( "Generating Graphs" )
            generate_graphs_cmd = "python3 generate_runtime_comparison_graphs.py " + " --sparkml_json " + abs_path_sparkml_json + \
                                  " --sparkling_water_json " + abs_path_sparkling_water_json + " --output_dir " + abs_path_sub_dir
            subprocess.run(generate_graphs_cmd, shell=True, check=True, env=prod_env)
        except:
            print("Generation of graphs failed")

        time.sleep(5)