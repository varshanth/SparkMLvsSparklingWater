import json
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils import events_summarizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparkml_json', type=str, required=True)
    parser.add_argument('--sparkling_water_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    return args

def plot_comparison_graph(spark_ml_event_info,
                          sparkling_water_event_info,
                          output_dir):
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
    plt.savefig(f'{output_dir}/setup_phase.png')

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
        plt.savefig(f'{output_dir}/{model_type}.png')

if __name__ == '__main__':
    args = get_args()
    spark_ml_json_path = args.sparkml_json
    sparkling_water_json_path = args.sparkling_water_json
    output_dir = args.output_dir

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

    spark_ml_events_summarizer = events_summarizer(spark_ml_events)
    sparkling_water_events_summarizer = events_summarizer(sparkling_water_events)

    spark_ml_model_types = spark_ml_events_summarizer.model_types
    sparkling_water_model_types = sparkling_water_events_summarizer.model_types

    if len(spark_ml_model_types) != len(sparkling_water_model_types):
        print('Model types covered are not the same')
        exit()

    spark_ml_events_summary = spark_ml_events_summarizer.events_summary
    sparkling_water_events_summary = sparkling_water_events_summarizer.events_summary

    plot_comparison_graph(spark_ml_events_summary,
            sparkling_water_events_summary,
            output_dir)
