import os
import glob

from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np


class DataSaver(object):

    def __init__(self, start_time_str=datetime.now().strftime("%m-%d-%Y-%H-%M-%S")):
        self.file_name = start_time_str + ".txt"
        self.file_path = "data/output" + self.file_name
        self._count = 0
        self.minimal_line_count = 5
        self.to_be_appended_lines = []
        print("Log file: ", self.file_path)

    def append(self, line):
        self.to_be_appended_lines.append(line)
        self._count += 1
        # Only create and write to file when
        if self._count > self.minimal_line_count:
            file = os.open(self.file_path, os.O_APPEND | os.O_RDWR | os.O_CREAT)
            for l in self.to_be_appended_lines:
                os.write(file, str.encode(l + '\n'))
            os.close(file)
            self.to_be_appended_lines = []


def create_average_reward_list(time_steps, sample_rewards, step_size, average_over_steps, scale_reward):
    """
    Creates a list of bins/points of average reward. The step_size defines how many log lines are used for each bin.
    An average is calculated over a certain amount of previous log lines. This is defined by average_over_log_lines.
    """
    sum_bin = 0
    count = 0
    current_step_bin = step_size
    average_rewards = []
    # create average points over
    last_reward_index = 0
    for i in range(len(time_steps)):
        sum_bin += sample_rewards[i]
        # remove oldest reward in rolling average
        if time_steps[i] - time_steps[last_reward_index] > average_over_steps:
            sum_bin -= sample_rewards[last_reward_index]
            last_reward_index += 1
        else:
            count += 1
        # Append rolling average to average_reward list
        if time_steps[i] > current_step_bin:
            current_step_bin += step_size
            average_rewards.append(sum_bin / count)
    return np.array(average_rewards) * scale_reward


def read_file(name_of_file):
    if name_of_file is None:
        list_of_files = glob.glob('data/*.txt')  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        name_of_file = latest_file
    else:
        name_of_file = "data/" + name_of_file
    print(name_of_file)
    file = open(name_of_file, 'r')  # use latest file or specify own file
    file.readline()  # skip first two lines with run info
    file.readline()
    time_steps = []
    rewards = []
    line_data = None
    for line in file:
        line_data = line.split(',')
        # Timestep
        time_steps.append(float(line_data[1]))
        # Rewards
        rewards.append(float(line_data[4]))
    return time_steps, rewards, float(line_data[-2])


def plot_average(average_reward_list, last_time_step, average_over_last_steps, x_label_additional_info="", show=True, color=None):
    steps = len(average_reward_list)
    time_steps = (last_time_step / steps) * np.arange(steps)
    print("Final reward: ", average_reward_list[-1])
    plt.plot(time_steps, average_reward_list, color)
    plt.xlabel(f"rewards averaged over last {average_over_last_steps} steps" + x_label_additional_info)
    plt.ylabel("mean rewards")
    if show:
        plt.show()


def plot_data(name_of_file=None, calculate_average_each_step=20000, average_over_last_steps=200000,
              scale_reward=100, only_average=True, show=True, color=""):
    time_steps, rewards, time_elapsed = read_file(name_of_file)

    print(f"time_elapsed: {int(float(time_elapsed))} seconds or {int(float(time_elapsed) / 60)} minutes")
    if not only_average:
        plt.plot(time_steps, rewards)
        plt.xlabel("timesteps")
        plt.ylabel("mean rewards")
        plt.show()

    average_reward_list = create_average_reward_list(time_steps, rewards, calculate_average_each_step,
                                                     average_over_last_steps, scale_reward)

    plot_average(average_reward_list, time_steps[-1], average_over_last_steps, show=show, color=color)
    return time_elapsed


def plot_runs(outputs, calculate_average_each_step=20000, average_over_last_steps=200000,
              scale_reward=100, show=True, color=""):
    average_lists = []
    time_steps = []
    last_run_length = 0
    total_time_list = []
    for file_name in outputs:
        time_steps, rewards, time_elapsed = read_file(file_name)
        average_reward_list = create_average_reward_list(time_steps, rewards, calculate_average_each_step,
                                                         average_over_last_steps, scale_reward)
        average_lists.append(average_reward_list)
        last_run_length = len(average_reward_list)
        total_time_list.append(time_elapsed)
    for i in range(len(average_lists)):
        average_lists[i] = average_lists[i][:last_run_length]

    total_time_list = np.array(total_time_list)
    average_lists = np.array(average_lists)
    average_of_all_runs = np.mean(average_lists, axis=0)
    variance_of_all_runs = np.var(average_lists, axis=0)
    last_time_step = time_steps[-1]

    steps = len(average_of_all_runs)
    time_steps = (last_time_step / steps) * np.arange(steps)
    lower_bound = average_of_all_runs - variance_of_all_runs
    upper_bound = average_of_all_runs + variance_of_all_runs
    plt.fill_between(time_steps, lower_bound, upper_bound, alpha=0.3, color=color)

    plot_average(average_of_all_runs, last_time_step, average_over_last_steps, f" over {len(average_lists)} runs", show, color=color)
    average_time_elapsed = np.mean(total_time_list)
    return average_time_elapsed


if __name__ == "__main__":
    plot_data("output03-31-2021-20-43-01.txt", average_over_last_steps=1000)
