#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Getting CLI args
parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', help='display the plot on screen')
parser.add_argument('--title', default='Compute vs. Communication Plot', help='set the title')
parser.add_argument('--eps', action='store_true', help='output .eps file')
parser.add_argument('--pdf', action='store_true', help='output .pdf file')
parser.add_argument('--png', action='store_true', help='output .png file')
parser.add_argument('--svg', action='store_true', help='output .svg file')
args = parser.parse_args()

# Parsing the data file
file_name='app1'
with open(file_name + '.csv') as f:
    names = f.readline().split(',')
    print(names)
    print("loading")
    data = np.loadtxt(f, delimiter=',', skiprows=1).transpose()
    print("done loading")
    time = data[1]/1000 # ms -> s
    compute = data[3]
    mpi = data[4]
    total = compute + mpi

    normalized_compute = np.divide(compute, total)
    normalized_mpi = np.divide(mpi, total)
    #normalized_compute = compute / total
    #normalized_mpi = mpi / total
    #time, normalized = data[0], np.divide(data[1:-1], data[-1])

print(normalized_compute)

# Plot formatting
plt.xlabel('Time (s)')
plt.ylabel('Percentage')
plt.gca().yaxis.set_visible(True)

plt.xlim(time[0], time[-1])
plt.ylim(0, 1)
plt.title(args.title)
plt.stackplot(time, normalized_compute, normalized_mpi, labels=['Compute', 'MPI'])
plt.legend(loc='lower right')

# Saving
if args.eps: plt.savefig(file_name + '.eps')
if args.pdf: plt.savefig(file_name + '.pdf')
if args.png: plt.savefig(file_name + '.png')
if args.svg: plt.savefig(file_name + '.svg')

if args.show:
    plt.show()
