import time
import json
from pyspark.sql import SparkSession

class logger:
    def __init__(self, output_json):
        self.output_json = output_json
        self.start_time = None
        self.events = []

    def start_timer(self):
        if self.start_time:
            print('Error: Timer already started!')
        else:
            self.start_time = time.time()
            self.log_event('Start', None)

    def stop_timer(self):
        if self.start_time:
            self.log_event('End', None)
            self.start_time = None
            if self.output_json:
                with open(self.output_json, 'w') as f:
                    json.dump(self.events, f, indent = 4)

        else:
            print('Error: Timer not started!')

    def log_event(self, event, value = None):
        current_time_obj = time
        current_time_epoch = current_time_obj.time()
        event_info = {
                'event' : event,
                'value' : value,
                'execution_time' : 0,
                'start_time' : current_time_obj.strftime('%Y-%m-%d %H:%M:%S'),
                'start_time_epoch' : current_time_epoch,
                }
        if len(self.events) == 0:
            self.events.append(event_info)
            return

        previous_event = self.events.pop()
        previous_event['execution_time'] = current_time_epoch - previous_event['start_time_epoch']
        previous_event['execution_time'] = 0 if previous_event['execution_time'] < 0.01 else previous_event['execution_time']
        self.events.append(previous_event)
        self.events.append(event_info)
        print(f"EXPERIMENT {event_info['start_time']} : ---- {event} = {value} ----")

class SparkConfig:
    def __init__(self, master_url, dataset, model_type):
        self.master_url = master_url
        self.dataset = dataset
        self.model_type = model_type
        self.worker_memory = "14g"
        self.executor_memory = "14g"
        self.rpc_message_maxSize = "1024"
        self.network_timeout = "300s"
        self.driver_memory = "14g"
        self.worker_cores = "10"
        self.executor_cores = "10"
        self.gc_interval = "10min"
        self.heartbeat_interval = "60s"

    def create_spark_session(self):
        spark = SparkSession.builder \
            .master(self.master_url) \
            .appName(f"PySpark_{self.dataset}_{self.model_type}") \
            .config("spark.worker.memory", self.worker_memory) \
            .config("spark.executor.memory", self.executor_memory) \
            .config("spark.rpc.message.maxSize", self.rpc_message_maxSize) \
            .config("spark.network.timeout", self.network_timeout) \
            .config("spark.driver.memory", self.driver_memory) \
            .config("spark.worker.cores", self.worker_cores) \
            .config("spark.executor.cores", self.executor_cores) \
            .config("spark.cleaner.periodicGC.interval", self.gc_interval) \
            .config("spark.executor.heartbeatInterval", self.heartbeat_interval) \
            .getOrCreate()

        return spark