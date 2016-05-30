import numpy as np
import matplotlib.pyplot as plt
import random
import math
from config import *

# CreateRandPath using Path Algorithm from reference
def CreatePath3(t):
    # initloc = np.array([MazeSize//2, MazeSize//4])
    initloc = np.array([25,25])
    path = np.zeros([t, 2])
    path[0,:] = initloc
    x = initloc[0]
    y = initloc[1]
    s = 2
    # m = 0.9
    # m = 0
    delx = 0
    dely = 0
    # index = 1

    # xy = 45

    path1 = np.zeros([180, 2])
    for i in range(0,50):
        path1[i,0] = 25 + 2*i
        path1[i,1] = 25
    # path1[50:60] = np.array([125, 25])
    for i in range(50,60):
        path1[i,0] = 125 + (random.random()-0.5)*10
        path1[i,1] = 25 + (random.random()-0.5)*10 

    for i in range(60,110):
        path1[i,0] = 125-(i-60)
        path1[i,1] = 25 + math.sqrt(3)*(i-60)
    # path1[110:120] = np.array([75, 25 + math.sqrt(3) * 50])
    for i in range(110,120):
        path1[i,0] = 75 + (random.random()-0.5)*10
        path1[i,1] = 25 + math.sqrt(3) * 50 + (random.random()-0.5)*10 

    for i in range(120,170):
        path1[i,0] = 75-(i-120)
        path1[i,1] = 25 +math.sqrt(3)*(170 - i)
    # path1[170:180] = np.array([25, 25])
    for i in range(170,180):
        path1[i,0] = 25 + (random.random()-0.5)*10
        path1[i,1] = 25 + (random.random()-0.5)*10 

    index = 0
    while index < t:
        tt = index % 180
        path[index,0] = path1[tt,0] + (random.random()-0.5)*3
        path[index,1] = path1[tt,1] + (random.random()-0.5)*3
        index += 1
    return path

def CreatePath4(t):
    # initloc = np.array([MazeSize//2, MazeSize//4])
    initloc = np.array([25,25])
    path = np.zeros([t, 2])
    path[0,:] = initloc
    x = initloc[0]
    y = initloc[1]
    s = 2
    # m = 0.9
    # m = 0
    delx = 0
    dely = 0
    # index = 1

    # xy = 45

    path1 = np.zeros([240, 2])
    for i in range(0,50):
        path1[i,0] = 25 + 2*i
        path1[i,1] = 25
    # path1[50:60] = np.array([125, 25])
    for i in range(50,60):
        path1[i,0] = 125 + (random.random()-0.5)*10
        path1[i,1] = 25 + (random.random()-0.5)*10 

    for i in range(60,110):
        path1[i,0] = 125
        path1[i,1] = 25 + 2*(i-60)
    # path1[110:120] = np.array([75, 25 + math.sqrt(3) * 50])
    for i in range(110,120):
        path1[i,0] = 125 + (random.random()-0.5)*10
        path1[i,1] = 125 + (random.random()-0.5)*10 

    for i in range(120,170):
        path1[i,0] = 125 - 2*(i-120)
        path1[i,1] = 125 
    # path1[170:180] = np.array([25, 25])
    for i in range(170,180):
        path1[i,0] = 25 + (random.random()-0.5)*10
        path1[i,1] = 125 + (random.random()-0.5)*10 

    for i in range(180,230):
        path1[i,0] = 25
        path1[i,1] = 125 - 2*(i-180) 
    # path1[170:180] = np.array([25, 25])
    for i in range(230,240):
        path1[i,0] = 25 + (random.random()-0.5)*10
        path1[i,1] = 25 + (random.random()-0.5)*10 

    index = 0
    while index < t:
        tt = index % 240
        path[index,0] = path1[tt,0] + (random.random()-0.5)*3
        path[index,1] = path1[tt,1] + (random.random()-0.5)*3
        index += 1
    return path

# path = CreatePath4(1000)
# plt.plot(path[:,0],path[:,1])
# path = CreatePath3(1000)
# plt.plot(path[:,0],path[:,1])
# plt.show()

