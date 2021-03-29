import os
import glob

from datetime import datetime
from matplotlib import pyplot as plt

class DataSaver(object):

    def __init__(self):

        self.file = "data/output"+datetime.now().strftime("%m-%d-%Y-%H-%M-%S")+".txt"
        open(self.file, 'w')

    def append(self, line):
        file = os.open(self.file, os.O_APPEND | os.O_RDWR | os.O_CREAT)
        os.write(file, str.encode(line + '\n'))
        os.close(file)

def main():

    list_of_files = glob.glob('data/*.txt')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    file = open(latest_file, 'r') # use latest file or specify own file
    file.readline() # skip first line with run info
    x = []
    y = []
    for line in file:
        line_data = line.split(',')
        x.append(line_data[1])
        y.append(line_data[4])

    plt.plot(x,y)
    plt.xlabel("timesteps")
    plt.ylabel("mean rewards")
    plt.show()

if __name__ == "__main__":
    main()