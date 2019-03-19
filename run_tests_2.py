import subprocess, time
import pdb

prod_env = env = {'SPARK_HOME': '/home/s6singla/spark-2.4.0-bin-hadoop2.7', 'TERM': 'xterm', 'SHELL': '/bin/bash', 'DERBY_HOME': '/usr/lib/jvm/java-8-oracle/db', 'SSH_CLIENT': '172.16.39.182 55252 22', 'CONDA_SHLVL'\
: '1', 'CONDA_PROMPT_MODIFIER': '(base) ', 'SSH_TTY': '/dev/pts/0', 'USER': 's6singla', 'LS_COLORS': 'rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=3\
7;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:\
*.t7z=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;\
31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.jpg=01;35:*.jpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;\
35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm\
=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=0\
1;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=\
00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:', 'CONDA_EXE': '/home/s6singla/anaconda3/bin/conda', 'MAIL': '/var/mail/s6singla', 'PATH': '/home/s6singla/bin:/home/s6singl\
a/.local/bin:/home/s6singla/anaconda3/bin:/home/s6singla/anaconda3/condabin:/home/s6singla/yes/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/l\
ib/jvm/java-8-oracle/bin:/usr/lib/jvm/java-8-oracle/db/bin:/usr/lib/jvm/java-8-oracle/jre/bin', 'CONDA_PREFIX': '/home/s6singla/anaconda3', 'PWD': '/home/s6singla', 'JAVA_HOME': '/usr/lib/jvm/java-8-oracl\
e', 'LANG': 'en_US.UTF-8', 'SHLVL': '1', 'HOME': '/home/s6singla', 'CONDA_PYTHON_EXE': '/home/s6singla/anaconda3/bin/python', 'LOGNAME': 's6singla', 'J2SDKDIR': '/usr/lib/jvm/java-8-oracle', 'XDG_DATA_DIR\
S': '/usr/local/share:/usr/share:/var/lib/snapd/desktop', 'SSH_CONNECTION': '172.16.39.182 55252 129.97.173.68 22', 'CONDA_DEFAULT_ENV': 'base', 'LESSOPEN': '| /usr/bin/lesspipe %s', 'DISPLAY': 'localhost\
:10.0', 'SPARKLING_HOME': '/home/s6singla/sparkling-water', 'J2REDIR': '/usr/lib/jvm/java-8-oracle/jre', 'LESSCLOSE': '/usr/bin/lesspipe %s %s', '_': '/home/s6singla/anaconda3/bin/python3'}

chunsize_cats_dogs_cmd = "--chunksize 100"
chunsize_susy_cmd = "--chunksize 10000"

pyspark_run_cmd = "spark-submit --conf spark.network.timeout=10000000 --driver-memory=14G --executor-memory=14G --num-executors=12 --executor-cores=12 test_sparkml.py"
pysparkling_run_cmd = "$SPARKLING_HOME/bin/run-python-script.sh test_sparkling_water.py"

model_lr_cmd = "--model_type logistic_regression"
model_kmeans_cmd = "--model_type kmeans"
model_pca_cmd = "--model_type pca"
model_mlp_cmd = "--model_type mlp"
model_all_cmd = "--model_type all"

path_to_csv_cats_dogs_cmd = "--path_to_csv ~/datasets/cats_dogs.csv"
path_to_csv_susy_cmd = "--path_to_csv ~/datasets/SUSY.csv"

dataset_cats_dogs_cmd = "--dataset cats_dogs"
dataset_susy_cmd = "--dataset susy"

output_dir = "/home/s6singla/SparkMLvsSparklingWater/output_dir_2/"

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

    '''
    for algo in ['lr', 'kmeans','pca', 'mlp']:
        if algo == 'lr':
            model_cmd = model_lr_cmd
        elif algo == 'kmeans':
            model_cmd = model_kmeans_cmd
        elif algo == 'pca':
            model_cmd = model_pca_cmd
        elif algo == 'mlp':
            model_cmd = model_mlp_cmd
    '''
    cmd = model_all_cmd + " " + path_to_csv_cmd + " " + dataset_cmd + " " + "--num_train_chunks=" + str(train_chunks) + " --num_test_chunks=" + str(test_chunks) +\
        " " + chunksize_cmd


    pyspark_cmd = pyspark_run_cmd + " " + cmd + " >> " + output_dir + output_file
    pysparkling_cmd = pysparkling_run_cmd + " " + cmd + " >> " + output_dir + output_file

    try:
        print( "Running cmd", pyspark_cmd )
        subprocess.run(pyspark_cmd, shell=True, check=True, env=prod_env)
    except:
        print("Pyspark Failed for chunk ", chunk)

    time.sleep(2)

    try:
        print( "Running cmd", pysparkling_cmd )
        subprocess.run(pysparkling_cmd, shell=True, check=True, env=prod_env)
    except:
        print("Pysparkling Failed for chunk ", chunk)