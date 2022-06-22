import os
import time

def make_dir(path):
    """
    """

    if not os.path.exists(path):
        os.makedirs(path)

def epoch_time(start_time, end_time):
    """
    """
    
    time_taken = end_time - start_time
    minutes = int(time_taken/60)
    seconds = int(time_taken - (minutes*60))

    return minutes, seconds
