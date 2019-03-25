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

class events_summarizer:
    def __init__(self, events):
        LOADING_DS_IDX = 7
        library_type_setup_idx = {
                'PySparkML': [LOADING_DS_IDX + 1 + i for i in range(3)],
                'PySparkling Water': [LOADING_DS_IDX + 1 + i for i in range(4)]
                }

        library_type = events[0]['value']
        if library_type not in list(library_type_setup_idx.keys()):
            print('Library Type Not Accepted')
        self.library_type = library_type
        self.setup_idx = library_type_setup_idx[self.library_type]
        self.events = events
        self.model_types = self._get_all_model_types()
        self.events_summary = self._get_events_summary()

    def get_setup_event_exec_time(self):
        event_exec_times = {}
        for idx in self.setup_idx:
            event_exec_times[self.events[idx]['event']] = self.events[idx]['execution_time']
        return event_exec_times

    def _get_all_model_types(self):
        model_types = []
        for event in self.events:
            if event['event'] == 'Training':
                model_types.append(event['value'])
        return model_types

    def get_model_start_end_idx(self, model_type):
        start_idx = end_idx = -1
        for idx, event in enumerate(self.events):
            if event['event'] == 'Training' and event['value'] == model_type:
                # Training event matching the model type
                start_idx = idx
            elif start_idx != -1 and event['event'] == 'Result':
                # Result of the event if the start idx was set already
                end_idx = idx
                break
        return start_idx, end_idx

    def get_model_training_exec_time(self, model_type):
        start_idx, end_idx = self.get_model_start_end_idx(model_type)
        if end_idx - start_idx == 1:
            # Training Failed
            return start_idx, -1
        return start_idx, self.events[start_idx]['execution_time']

    def get_model_testing_exec_time(self, model_type):
        start_idx, end_idx = self.get_model_start_end_idx(model_type)
        if end_idx - start_idx == 1:
            # Training Failed
            return -1, -1
        elif not self.events[end_idx]['value']:
            # Testing Failed
            return -1, -1
        # Find Testing event
        for idx, event in enumerate(self.events[start_idx+1:end_idx]):
            if event['event'] == 'Testing':
                return start_idx+1+idx, event['execution_time']
        # Testing Event Not Found even though result is True
        return -1, -1

    def get_model_event_from_existing_range(self, start_idx, end_idx, event_type):
        for idx, event in enumerate(self.events[start_idx:end_idx]):
            if event['event'] == event_type:
                return start_idx+idx, event['value']
        return -1, -1

    def _get_events_summary(self):
        # Populate setup phase execution times
        events_summary = {'setup' : self.get_setup_event_exec_time()}

        # Populate model information
        for model_type in self.model_types:
            train_idx, train_exec_time = self.get_model_training_exec_time(model_type)
            test_idx, test_exec_time = self.get_model_testing_exec_time(model_type)

            events_summary[model_type] = {
                    'exec_time': (train_exec_time, test_exec_time)
                    }
            model_start_idx, model_end_idx = self.get_model_start_end_idx(model_type)
            idx, value = self.get_model_event_from_existing_range(
                    model_start_idx, model_end_idx, 'Training Accuracy')
            events_summary[model_type]['training_accuracy'] = float(value)
            idx, value = self.get_model_event_from_existing_range(
                        model_start_idx, model_end_idx, 'Testing Accuracy')
            events_summary[model_type]['testing_accuracy'] =  float(value)
        return events_summary

