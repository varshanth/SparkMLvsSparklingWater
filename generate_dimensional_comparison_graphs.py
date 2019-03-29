import json
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils import events_summarizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_title', type=str, required=True)
    parser.add_argument('--dimension_type', type=str, required=True)
    parser.add_argument('--dimensions', type=int, nargs='+', required=True)
    parser.add_argument('--result_type', type=str,
            choices = ['Runtime(Sec)', 'Accuracy'], required=True)
    parser.add_argument('--sparkml_jsons', type=str, nargs='+', required=True)
    parser.add_argument('--sparkling_water_jsons', type=str, nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    return args

def plot_comparison_graph(graph_title,
                          dimension_type,
                          dimensions,
                          result_type,
                          spark_ml_events_summaries,
                          sparkling_water_events_summaries,
                          output_dir):

    num_summaries = len(spark_ml_events_summaries)
    width = 0.35
    ind_sparkml = np.arange(1, num_summaries+1)
    ind_sparkling_water = np.arange(1, num_summaries+1) + width
    ind_merged = np.sort(np.append(ind_sparkml, ind_sparkling_water))
    xticks_spml = [f'{dim}_SpML' for dim in dimensions]
    xticks_sw = [f'{dim}_SW' for dim in dimensions]
    xticks_merged = [merged[i] for merged in list(zip(xticks_spml, xticks_sw))
            for i in range(len(merged))]
    acc_yticks = [round(i,1) for i in np.arange(0.0, 1.1, 0.1)]

    if result_type == 'Runtime(Sec)':
        # Plot Setup Phase
        p_bars = []
        spark_ml_events = set(list(spark_ml_events_summaries[0]['setup'].keys()))
        sparkling_water_events = set(list(sparkling_water_events_summaries[0]['setup'].keys()))
        union_events = spark_ml_events.union(sparkling_water_events)
        intersect_events = spark_ml_events.intersection(sparkling_water_events)
        # Display common events on the bottom
        unique_events = list(intersect_events)+list(union_events-intersect_events)

        previous_results = np.zeros(num_summaries*2)
        for unique_event in unique_events:
            result_sparkml = [0 for i in range(num_summaries)]
            result_sparkling_water = [0 for i in range(num_summaries)]
            if unique_event in spark_ml_events:
                result_sparkml = [spark_ml_event_info['setup'][unique_event]
                        for spark_ml_event_info in spark_ml_events_summaries]
            if unique_event in sparkling_water_events:
                result_sparkling_water = [sparkling_water_event_info['setup'][unique_event]
                        for sparkling_water_event_info in sparkling_water_events_summaries]
            result = np.array([merged[i] for merged in list(zip(result_sparkml,
                result_sparkling_water)) for i in range(len(merged))])
            p_bars.append(plt.bar(ind_merged, result, width, bottom=previous_results))
            previous_results += result
        plt.xlabel(dimension_type)
        plt.ylabel(result_type)
        plt.title(f'{graph_title}: Setup Phase')
        plt.xticks(ind_merged, xticks_merged)
        plt.legend((p_bar[0] for p_bar in p_bars), (unique_event for unique_event in unique_events))
        plt.savefig(f'{output_dir}/setup.png')

    # Plot Model Specific Comparisons
    model_types = [key for key in spark_ml_events_summaries[0].keys() if key != 'setup']

    for model_type in model_types:
        if result_type == 'Runtime(Sec)':
            spark_ml_exec_times = [spark_ml_event_info[model_type]['exec_time']
                    for spark_ml_event_info in spark_ml_events_summaries]
            sparkling_water_exec_times = [sparkling_water_event_info[model_type]['exec_time']
                    for sparkling_water_event_info in sparkling_water_events_summaries]

            # Plot Training Times
            plt.close()
            # 2 Different bars to ensure color difference
            sparkml_bar = plt.bar(ind_sparkml,
                    [ex_time[0] for ex_time in spark_ml_exec_times], width)
            sparkling_water_bar = plt.bar(ind_sparkling_water,
                    [ex_time[0] for ex_time in sparkling_water_exec_times], width)
            plt.xlabel(dimension_type)
            plt.ylabel(result_type)
            plt.title(f'{graph_title}: Training Duration for {model_type}')
            plt.xticks(ind_merged, xticks_merged)
            plt.legend([sparkml_bar, sparkling_water_bar], ['SparkML', 'Sparkling Water'])
            plt.savefig(f'{output_dir}/{model_type}_train_time.png')

            #Plot Testing Times
            plt.close()
            # 2 Different bars to ensure color difference
            sparkml_bar = plt.bar(ind_sparkml,
                    [ex_time[1] for ex_time in spark_ml_exec_times], width)
            sparkling_water_bar = plt.bar(ind_sparkling_water,
                    [ex_time[1] for ex_time in sparkling_water_exec_times], width)
            plt.xlabel(dimension_type)
            plt.ylabel(result_type)
            plt.title(f'{graph_title}: Testing Duration for {model_type}')
            plt.xticks(ind_merged, xticks_merged)
            plt.legend([sparkml_bar, sparkling_water_bar], ['SparkML', 'Sparkling Water'])
            plt.savefig(f'{output_dir}/{model_type}_test_time.png')

        elif result_type == 'Accuracy':
            spark_ml_accuracies = [spark_ml_event_info[model_type]['accuracy']
                    for spark_ml_event_info in spark_ml_events_summaries]
            sparkling_water_accuracies = [sparkling_water_event_info[model_type]['accuracy']
                    for sparkling_water_event_info in sparkling_water_events_summaries]

            # Plot Training Accuracies
            plt.close()
            # 2 Different bars to ensure color difference
            sparkml_bar = plt.bar(ind_sparkml,
                    [acc[0] for acc in spark_ml_accuracies], width)
            sparkling_water_bar = plt.bar(ind_sparkling_water,
                    [acc[0] for acc in sparkling_water_accuracies], width)
            plt.xlabel(dimension_type)
            plt.ylabel(result_type)
            plt.title(f'{graph_title}: Training Accuracy for {model_type}')
            plt.xticks(ind_merged, xticks_merged)
            plt.legend([sparkml_bar, sparkling_water_bar], ['SparkML', 'Sparkling Water'])
            plt.savefig(f'{output_dir}/{model_type}_train_acc.png')

            #Plot Testing Accuracies
            plt.close()
            # 2 Different bars to ensure color difference
            sparkml_bar = plt.bar(ind_sparkml,
                    [acc[1] for acc in spark_ml_accuracies], width)
            sparkling_water_bar = plt.bar(ind_sparkling_water,
                    [acc[1] for acc in sparkling_water_accuracies], width)
            plt.xlabel(dimension_type)
            plt.ylabel(result_type)
            plt.title(f'{graph_title}: Testing Accuracy for {model_type}')
            plt.xticks(ind_merged, xticks_merged)
            plt.legend([sparkml_bar, sparkling_water_bar], ['SparkML', 'Sparkling Water'])
            plt.savefig(f'{output_dir}/{model_type}_test_acc.png')

        else:
            print(f'{result_type} not implemented!')


if __name__ == '__main__':
    args = get_args()
    graph_title = args.graph_title
    dimension_type = args.dimension_type
    dimensions = args.dimensions
    result_type = args.result_type
    spark_ml_jsons_path = args.sparkml_jsons
    sparkling_water_jsons_path = args.sparkling_water_jsons
    output_dir = args.output_dir

    # Check if number of SparkML JSONs equal to Sparkling Water JSONs
    assert len(spark_ml_jsons_path) == len(sparkling_water_jsons_path), \
            'Number of JSONs of each type should be equal'

    assert len(args.dimensions) == len(spark_ml_jsons_path), \
            'Number of dimensions should match number of JSONs'

    spark_ml_events_list = []
    sparkling_water_events_list = []
    ''' Might Need Later
    spark_ml_events_summarizer_list = []
    sparkling_water_events_summarizer_list = []
    '''
    spark_ml_events_summaries = []
    sparkling_water_events_summaries = []


    for spark_ml_json_path in spark_ml_jsons_path:
        with open(spark_ml_json_path, 'r') as f:
            spark_ml_events = json.load(f)
            spark_ml_events_list.append(spark_ml_events)
    for sparkling_water_json_path in sparkling_water_jsons_path:
        with open(sparkling_water_json_path, 'r') as f:
            sparkling_water_events = json.load(f)
            sparkling_water_events_list.append(sparkling_water_events)

    for joint_events in zip(spark_ml_events_list, sparkling_water_events_list):
        spark_ml_events, sparkling_water_events = joint_events

        # Check if JSONs are correct
        spark_ml_library = spark_ml_events[0]['value']
        sparkling_water_library = sparkling_water_events[0]['value']
        if spark_ml_library != 'PySparkML' or \
                sparkling_water_library != 'PySparkling Water':
            print('JSONs are not of correct libraries')
            exit()

        # Retrieve Event Summaries
        spark_ml_events_summarizer = events_summarizer(spark_ml_events)
        sparkling_water_events_summarizer = events_summarizer(
                sparkling_water_events)

        # Check if model types match
        spark_ml_model_types = spark_ml_events_summarizer.model_types
        sparkling_water_model_types = sparkling_water_events_summarizer.model_types

        if len(spark_ml_model_types) != len(sparkling_water_model_types):
            print('Model types covered are not the same')
            exit()

        ''' Might Need Later
        # Save event summarizers
        spark_ml_events_summarizer_list.append(spark_ml_events_summarizer)
        sparkling_water_events_summarizer_list.append(events_summarizer(
            sparkling_water_events))
        '''

        # Save event summaries
        spark_ml_events_summaries.append(
                spark_ml_events_summarizer.events_summary)
        sparkling_water_events_summaries.append(
                sparkling_water_events_summarizer.events_summary)

    plot_comparison_graph(
            graph_title,
            dimension_type,
            dimensions,
            result_type,
            spark_ml_events_summaries,
            sparkling_water_events_summaries,
            output_dir)
