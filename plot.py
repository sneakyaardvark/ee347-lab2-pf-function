import numpy as np
from matplotlib import pyplot as plt
from pfcorrect import correct
import csv

PF_COL = 3
S_COL = 4
Q_COL = 5

def plot():
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    _p_load = []
    _q_load = []
    _pf = []
    with open("data/data_all.csv", newline='') as csvf:
        rdr = csv.reader(csvf)
        for row in rdr:
            print(row)
            _pf.append(float(row[PF_COL]))
            _p_load.append(float(row[S_COL]))
            _q_load.append(float((row[Q_COL])))

    ax1.set(title='Before Compensation', xlabel='$P_{load}$', ylabel='$Q_{load}$', zlabel='PF')
    ax1.plot(_p_load, _q_load, _pf)

    ax2.set(title='After Compensation', xlabel='$P_{load}$', ylabel='$Q_{load}$', zlabel='PF')

    plt.show()
            
if __name__ == "__main__":
    plot()
