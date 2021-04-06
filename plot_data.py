import os
import glob

from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

class DataSaver(object):

    def __init__(self, start_time_str=datetime.now().strftime("%m-%d-%Y-%H-%M-%S")):
        self.file = "data/output" + start_time_str + ".txt"
        # open(self.file, 'w')
        self._count = 0
        self.minimal_line_count = 5
        self.to_be_appended_lines = []
        print("Log file: ", self.file)

    def append(self, line):
        self.to_be_appended_lines.append(line)
        self._count += 1
        # Only create and write to file when
        if self._count > self.minimal_line_count:
            file = os.open(self.file, os.O_APPEND | os.O_RDWR | os.O_CREAT)
            for l in self.to_be_appended_lines:
                os.write(file, str.encode(l + '\n'))
            os.close(file)
            self.to_be_appended_lines = []


def create_average_reward_list(x_list, y_list, step_size=100, average_size=10000):
    sum_bin = 0
    count = 0
    current_step_bin = step_size
    average_reward = []
    # remove_first = y_list[0]
    # create average points over
    for i in range(len(x_list)):
        count += 1
        sum_bin += y_list[i]
        if x_list[i] > current_step_bin:
            current_step_bin += step_size
            average_reward.append(sum_bin / count)
            if i > average_size:
                sum_bin -= y_list[i-average_size]
                count = average_size - 1
    return average_reward


def plot_data(name_of_file=None):
    if name_of_file is None:
        list_of_files = glob.glob('data/*.txt')  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        name_of_file = latest_file
    else:
        name_of_file = "data/"+name_of_file
    print(name_of_file)
    file = open(name_of_file, 'r')  # use latest file or specify own file
    file.readline()  # skip first two lines with run info
    file.readline()
    x = []
    y = []
    time_elapsed = 0
    for line in file:
        line_data = line.split(',')
        # Timestep
        x.append(float(line_data[1]))
        # Rewards
        y.append(float(line_data[4]))
        time_elapsed = line_data[-2]
    print(f"time_elapsed: {int(float(time_elapsed))} seconds or {int(float(time_elapsed)/60)} minutes")
    average_over_steps = 10000
    calculate_average_each_step = 100

    plt.plot(x, y)
    plt.xlabel("timesteps")
    plt.ylabel("mean rewards")
    plt.show()

    average_reward_list = create_average_reward_list(x, y, calculate_average_each_step ,average_over_steps)
    x = average_over_steps*np.arange(len(average_reward_list))
    plt.plot(x, average_reward_list)
    plt.xlabel(f"timesteps averaged over {average_over_steps}")
    plt.ylabel("mean rewards")
    plt.show()


if __name__ == "__main__":
    plot_data()
