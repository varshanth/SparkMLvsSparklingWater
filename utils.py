from datetime import datetime

def log_with_time(print_msg):
    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"EXPERIMENT {time_now}: {print_msg}")

