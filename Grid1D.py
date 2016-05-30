import numpy as np
import matplotlib.pyplot as plt
import random
from config import *
from scipy.signal import argrelextrema

def CreateGrid (GridNum,lam):
    "Using three 2-D Cosine Function Models"
    "Spcing is the field distance of grid cells"
    "Min spacing is 0.2; medium is 0.7; max is 1.2;"
    spacing = np.linspace(20,max_lamda,GridNum)
    if GridNum == 1:
        spacing[0] = lam
    last = np.zeros([MazeSize, GridNum])
    random.seed(10)

    for k in range(0, GridNum):
        # phasex = 0
        phasex = int(MazeSize/2)
        for i in range(0,MazeSize):
            # last[i,k] =  1+ np.cos(2*np.pi/spacing[k] * i + phasex)
            last[i,k] =  np.exp((np.cos(2*np.pi/spacing[k] * (i - phasex))-1))
    return last

def cross_correlation(cells1,cells2):
    k1 = cells1.shape[1]
    k2 = cells2.shape[1]
    mul2 = np.zeros((k1,k2))
    for i in range(0,k1):
        for j in range(0,k2):
            mul2 [i,j] = np.correlate(cells1[:,i].reshape(-1), cells2[:,j].reshape(-1))
    mul2 = mul2/mul2.max()
    return mul2

MazeSize = 300
GridNum = 500
max_lamda = 200
grids = CreateGrid(GridNum,0)
x = np.linspace(20,max_lamda,GridNum)
# for i in [30,40,60,80,100, 120]:
for i in [40]:
    grids1 = CreateGrid(1,i)
    plt.plot(grids1)
    cor = cross_correlation(grids1,grids)
    local_max_index = argrelextrema(cor.reshape(-1),np.greater)[0]

    print (x[local_max_index])
    plt.figure()
    plt.plot(x[local_max_index]/i)
    print (x[local_max_index]/i)
    # xx = np.arange(0,len(local_max_index))
    # yy = x[local_max_index]/i
    # z = np.polyfit(xx, yy, 2)
    # print (z)

    plt.figure()
    plt.plot(x,cor[0,:])
    plt.plot(x[local_max_index], cor[0,local_max_index], "o", label="min")
    plt.xlabel('lamda (cm)')
    plt.ylabel('Normalized Cross-Correlation')

    diff = [(x[local_max_index][i])/x[local_max_index][i-1] for i in range(1, len(local_max_index))]
    # diff = [(x[local_max_index][j]/x[local_max_index][j-1]) for j in range(1, len(local_max_index))]
    # print (diff)
    ave = [np.average(diff) for i in range(20, max_lamda)]
    print (np.average(diff))
    plt.figure()
    plt.scatter(x[local_max_index][1:],diff)
    plt.plot(range(20, max_lamda),ave, color='r', linewidth=1.5)
    plt.xlabel('Peak Location (cm)')
    plt.xlim(20,200)
    plt.ylabel('Peak Ratio')

# for i in plt.get_fignums():
#     plt.figure(i)
#     plt.savefig('figures/Grid1D_%s.pdf' % i)
    # plt.savefig('figures/Sim4RewRan1000_%s.pdf' % i)
plt.show()

