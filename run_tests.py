import subprocess, time

chunsize_cats_dogs_cmd = "--chunksize 100"
chunsize_susy_cmd = "--chunksize 10000"

pyspark_run_cmd = "spark-submit test_sparkml.py"
pysparkling_run_cmd = "$SPARKLING_HOME/bin/run-python-script.sh test_sparkling_water.py"

model_lr_cmd = "--model_type logistic_regression"
model_kmeans_cmd = "--model_type kmeans"
model_pca_cmd = "--model_type pca"
model_mlp_cmd = "--model_type mlp"

path_to_csv_cats_dogs_cmd = "--path_to_csv ~/datasets/cats_dogs.csv"
path_to_csv_susy_cmd = "--path_to_csv ~/datasets/SUSY.csv"

dataset_cats_dogs_cmd = "--dataset cats_dogs"
dataset_susy_cmd = "--dataset susy"

chunks = []
chunks.append(('cats_dogs', 70, 12))
chunks.append(('susy', 180, 38))
chunks.append(('cats_dogs', 140, 24))
chunks.append(('susy', 360, 76))
chunks.append(('cats_dogs', 210, 36))
chunks.append(('susy', 540, 114))
chunks.append(('cats_dogs', 280, 48))
chunks.append(('susy', 180, 38))
chunks.append(('cats_dogs', 350, 60))
chunks.append(('susy', 360, 76))
chunks.append(('cats_dogs', 420, 72))
chunks.append(('susy', 540, 114))

for algo in ['lr', 'kmeans','pca', 'mlp']:
    if algo == 'lr':
        model_cmd = model_lr_cmd
    elif algo == 'kmeans':
        model_cmd = model_kmeans_cmd
    elif algo == 'pca':
        model_cmd = model_pca_cmd
    elif algo == 'mlp':
        model_cmd = model_mlp_cmd

    for chunk in chunks:
        dataset, train_chunks, test_chunks = chunk

        if dataset == 'cats_dogs':
            path_to_csv_cmd = path_to_csv_cats_dogs_cmd
            dataset_cmd = dataset_cats_dogs_cmd
            chunksize_cmd = chunsize_cats_dogs_cmd
        elif dataset == 'susy':
            path_to_csv_cmd = path_to_csv_susy_cmd
            dataset_cmd = dataset_susy_cmd
            chunksize_cmd = chunsize_susy_cmd

        output_file = dataset + "_" + str(train_chunks) + "_" + str(test_chunks)
        cmd = model_cmd + " " + path_to_csv_cmd + " " + dataset_cmd + " " + "--num_train_chunks=" + str(train_chunks) + " --num_test_chunks=" + str(test_chunks) +\
              " " + chunksize_cmd

        pyspark_run_cmd = pyspark_run_cmd + " " + cmd + " >> " + output_file
        pysparkling_run_cmd = pysparkling_run_cmd + " " + cmd + " >> " + output_file

        try:
            subprocess.run(pyspark_run_cmd, shell=True, check=True)
        except:
            print("Pyspark Failed for algo %s chunk %s", (algo, chunk))

        time.sleep(2)

        try:
            subprocess.run(pysparkling_run_cmd, shell=True, check=True)
        except:
            print("Pysparkling Failed for algo %s chunk %s", (algo, chunk))
