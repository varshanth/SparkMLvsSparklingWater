import time
import json


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


