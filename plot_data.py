import os
import glob

from datetime import datetime
from matplotlib import pyplot as plt


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


def create_average_reward_list(x_list, y_list, step_size):
    sum_bin = 0
    count = 0
    current_step_bin = step_size
    average_reward = []
    # create average points over
    for i in range(len(x_list)):
        count += 1
        sum_bin += y_list[i]
        if x_list[i] > current_step_bin:
            current_step_bin += step_size
            average_reward.append(sum_bin / count)
            sum_bin = 0
            count = 0
    return average_reward


def main():
    list_of_files = glob.glob('data/*.txt')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    file = open(latest_file, 'r')  # use latest file or specify own file
    file.readline()  # skip first two lines with run info
    file.readline()
    x = []
    y = []
    for line in file:
        line_data = line.split(',')
        # Timestep
        x.append(float(line_data[1]))
        # Rewards
        y.append(float(line_data[4]))

    average_over_steps = 10000

    plt.plot(x, y)
    plt.xlabel("timesteps")
    plt.ylabel("mean rewards")
    plt.show()
    average_reward_list = create_average_reward_list(x, y, average_over_steps)
    plt.plot(average_reward_list)
    plt.xlabel(f"timesteps averaged over {average_over_steps}")
    plt.ylabel("mean rewards")
    plt.show()


if __name__ == "__main__":
    main()
