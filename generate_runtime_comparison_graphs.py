import json
from sys import argv
import matplotlib.pyplot as plt
import numpy as np


LOADING_DS_IDX = 7
SPARKML_SETUP_IDX = [LOADING_DS_IDX + 1 + i for i in range(3)]
SPARKLING_WATER_SETUP_IDX = [LOADING_DS_IDX + 1 + i for i in range(4)]


def get_setup_event_exec_time(events):
    lib_type = events[0]['value']
    event_exec_times = {}
    if lib_type == 'PySparkML':
        setup_idx = SPARKML_SETUP_IDX
    else:
        setup_idx = SPARKLING_WATER_SETUP_IDX
    for idx in setup_idx:
        event_exec_times[events[idx]['event']] = events[idx]['execution_time']
    return event_exec_times

def get_all_model_types(events):
    model_types = []
    for event in events:
        if event['event'] == 'Training':
            model_types.append(event['value'])
    return model_types

def get_model_start_end_idx(events, model_type):
    start_idx = end_idx = -1
    for idx, event in enumerate(events):
        if event['event'] == 'Training' and event['value'] == model_type:
            # Training event matching the model type
            start_idx = idx
        elif start_idx != -1 and event['event'] == 'Result':
            # Result of the event if the start idx was set already
            end_idx = idx
            break
    return start_idx, end_idx

def get_model_training_exec_time(events, model_type):
    start_idx, end_idx = get_model_start_end_idx(events, model_type)
    if end_idx - start_idx == 1:
        # Training Failed
        return start_idx, -1
    return start_idx, events[start_idx]['execution_time']

def get_model_testing_exec_time(events, model_type):
    start_idx, end_idx = get_model_start_end_idx(events, model_type)
    if end_idx - start_idx == 1:
        # Training Failed
        return -1, -1
    elif not events[end_idx]['value']:
        # Testing Failed
        return -1, -1
    # Find Testing event
    for idx, event in enumerate(events[start_idx+1:end_idx]):
        if event['event'] == 'Testing':
            return start_idx+1+idx, event['execution_time']
    # Testing Event Not Found even though result is True
    return -1, -1

def get_model_event_from_existing_range(events, start_idx, end_idx, event_type):
    for idx, event in enumerate(events[start_idx:end_idx]):
        if event['event'] == event_type:
            return start_idx+idx, event['value']
    return -1, -1

def get_spark_ml_sparkling_water_event_info(spark_ml_events, sparkling_water_events):
    spark_ml_event_info = {}
    sparkling_water_event_info = {}

    # Populate setup phase execution times
    spark_ml_event_info['setup'] = get_setup_event_exec_time(spark_ml_events)

    sparkling_water_event_info['setup'] = get_setup_event_exec_time(sparkling_water_events)
    model_types = get_all_model_types(spark_ml_events)
    # Populate model information
    for model_type in model_types:
        # Spark ML
        train_idx, train_exec_time = get_model_training_exec_time(
                spark_ml_events, model_type)
        test_idx, test_exec_time = get_model_testing_exec_time(
                spark_ml_events, model_type)

        spark_ml_event_info[model_type] = {
                'exec_time': (train_exec_time, test_exec_time)
                }
        model_start_idx, model_end_idx = get_model_start_end_idx(
                spark_ml_events, model_type)
        idx, value = get_model_event_from_existing_range(
                spark_ml_events,
                model_start_idx, model_end_idx, 'Training Accuracy')
        spark_ml_event_info[model_type]['training_accuracy'] = float(value)
        idx, value = get_model_event_from_existing_range(spark_ml_events,
                    model_start_idx, model_end_idx, 'Testing Accuracy')
        spark_ml_event_info[model_type]['testing_accuracy'] =  float(value)

        # Sparkling Water
        train_idx, train_exec_time = get_model_training_exec_time(
                sparkling_water_events, model_type)
        test_idx, test_exec_time = get_model_testing_exec_time(
                sparkling_water_events, model_type)
        sparkling_water_event_info[model_type] = {
                'exec_time' : (train_exec_time, test_exec_time)
                }
        model_start_idx, model_end_idx = get_model_start_end_idx(
                sparkling_water_events, model_type)
        idx, value = get_model_event_from_existing_range(
                sparkling_water_events,
                model_start_idx, model_end_idx, 'Training Accuracy')
        sparkling_water_event_info[model_type]['training_accuracy'] = float(value)
        idx, value = get_model_event_from_existing_range(sparkling_water_events,
                    model_start_idx, model_end_idx, 'Testing Accuracy')
        sparkling_water_event_info[model_type]['testing_accuracy'] = \
                float(value)

    return spark_ml_event_info, sparkling_water_event_info

def plot_comparison_graph(spark_ml_event_info, sparkling_water_event_info):
    width = 0.35
    ind = (1, 2)
    xticks = ('SparkML', 'Sparkling Water')
    acc_yticks = [round(i,1) for i in np.arange(0.0, 1.1, 0.1)]
    # Plot Setup Phase
    p_bars = []
    spark_ml_events = set(list(spark_ml_event_info['setup'].keys()))
    sparkling_water_events = set(list(sparkling_water_event_info['setup'].keys()))
    union_events = spark_ml_events.union(sparkling_water_events)
    intersect_events = spark_ml_events.intersection(sparkling_water_events)
    # Display common events on the bottom
    unique_events = list(intersect_events)+list(union_events-intersect_events)
    previous_results = [0,0]
    for unique_event in unique_events:
        result = [0, 0]
        if unique_event in spark_ml_events:
            result[0] = spark_ml_event_info['setup'][unique_event]
        if unique_event in sparkling_water_events:
            result[1] = sparkling_water_event_info['setup'][unique_event]
        p_bars.append(plt.bar(ind, result, width, bottom=previous_results))
        previous_results = [previous_results[0]+result[0], previous_results[1]+result[1]]
    plt.ylabel('Run Time (sec)')
    plt.title('Setup Phase Run Time Comparison')
    plt.xticks(ind, xticks)
    plt.legend((p_bar[0] for p_bar in p_bars), (unique_event for unique_event in unique_events))
    plt.show()

    # Plot Model Specific Comparisons
    model_types = [key for key in spark_ml_event_info.keys() if key != 'setup']
    for model_type in model_types:
        plt.close()
        spark_ml_exec_times = spark_ml_event_info[model_type]['exec_time']
        sparkling_water_exec_times = sparkling_water_event_info[model_type]['exec_time']
        spark_ml_accuracies = (spark_ml_event_info[model_type]['training_accuracy'],
                spark_ml_event_info[model_type]['testing_accuracy'])
        sparkling_water_accuracies = (sparkling_water_event_info[model_type]['training_accuracy'],
                sparkling_water_event_info[model_type]['testing_accuracy'])
        print(model_type, spark_ml_accuracies, sparkling_water_accuracies)
        temp_acc = 1.0
        for i in range(2):
            temp_acc *= spark_ml_accuracies[i]
            temp_acc *= sparkling_water_accuracies[i]

        num_plt_rows = 1 if temp_acc == 1.0 else 2

        # Plot Training Times
        plt.subplot(num_plt_rows, 2, 1)
        result = (spark_ml_exec_times[0], sparkling_water_exec_times[0])
        plt.bar(ind, result, width)
        plt.ylabel('Run Time (sec)')
        plt.title(f'Training Time: {model_type}')
        plt.xticks(ind, xticks)

        #Plot Testing Times
        plt.subplot(num_plt_rows, 2, 2)
        result = (spark_ml_exec_times[1], sparkling_water_exec_times[1])
        plt.bar(ind, result, width)
        plt.ylabel('Run Time (sec)')
        plt.title(f'Testing Time: {model_type}')
        plt.xticks(ind, xticks)
        plt.tight_layout()

        if num_plt_rows == 2:
            #Plot Training Accuracies
            plt.subplot(num_plt_rows, 2, 3)
            result = (spark_ml_accuracies[0], sparkling_water_accuracies[0])
            plt.bar(ind, result, width)
            plt.ylabel('Accuracy')
            plt.title(f'Training Accuracy: {model_type}')
            plt.xticks(ind, xticks)
            plt.yticks(acc_yticks)
            plt.tight_layout()

            #Plot Testing Accuracies
            plt.subplot(num_plt_rows, 2, 4)
            result = (spark_ml_accuracies[1], sparkling_water_accuracies[1])
            plt.bar(ind, result, width)
            plt.ylabel('Accuracy')
            plt.title(f'Testing Accuracy: {model_type}')
            plt.xticks(ind, xticks)
            plt.yticks(acc_yticks)
            plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    spark_ml_json_path = argv[1]
    sparkling_water_json_path = argv[2]

    with open(spark_ml_json_path, 'r') as f:
        spark_ml_events = json.load(f)

    with open(sparkling_water_json_path, 'r') as f:
        sparkling_water_events = json.load(f)

    # Check if JSONs are correct
    spark_ml_library = spark_ml_events[0]['value']
    sparkling_water_library = sparkling_water_events[0]['value']

    if spark_ml_library != 'PySparkML' or \
            sparkling_water_library != 'PySparkling Water':
        print('JSONs are not of correct libraries')
        exit()

    spark_ml_model_types = get_all_model_types(spark_ml_events)
    sparkling_water_model_types = get_all_model_types(sparkling_water_events)

    if len(spark_ml_model_types) != len(sparkling_water_model_types):
        print('Model types covered are not the same')
        exit()

    spark_ml_event_info, sparkling_water_event_info = \
            get_spark_ml_sparkling_water_event_info(spark_ml_events,
                    sparkling_water_events)

    plot_comparison_graph(spark_ml_event_info, sparkling_water_event_info)
