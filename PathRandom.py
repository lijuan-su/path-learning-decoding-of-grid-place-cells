import numpy as np
import matplotlib.pyplot as plt
import random
import math
from config import *

# CreateRandPath using Path Algorithm from reference
def CreatePath(t,m):
    # initloc = np.array([MazeSize//2, MazeSize//4])
    initloc = np.array([75,10])
    path = np.zeros([t, 2])
    path[0,:] = initloc
    x = initloc[0]
    y = initloc[1]
    s = 5
    # m = 0.5
    # m = 0
    delx = 0
    dely = 0
    index = 1
    random.seed()

    while index < t:
        px = np.random.normal(0,1)
        py = np.random.normal(0,1)
        delx = s * (1-m) * px + m * delx
        dely = s * (1-m) * py + m * dely
        x = round(x + delx)
        y = round(y + dely)
        
        if (np.sqrt((x-MazeSize/2.0)*(x-MazeSize/2.0)+(y-MazeSize/2.0)*(y-MazeSize/2.0)) < MazeSize/2):
            path[index,:] = [x, y]
            index = index + 1
        else :
            delx = -1 * delx
            dely = -1 * dely
            x = round(x + delx)
            y = round(y + dely)
            path[index,:] = [x,y]
            index = index + 1
            # xy += 60

    return path
def path2maze(path):
    path_maze = np.zeros([MazeSize, MazeSize])
    for i in range(0,path.shape[0]):
        x = path[i, 0]
        y = path[i, 1]
        path_maze[x-1,y-1] += 1
    fig = plt.figure()
    cs = plt.imshow(path_maze/path_maze.max())
    # plt.title(title[title_i],fontsize=11)
    fig.colorbar(cs, shrink=0.7, pad=0.02)
    return path

# path = CreatePath(1000,0)
# plt.figure()
# plt.plot(path[:,0],path[:,1])

# path = CreatePath(1000,0.5)
# plt.figure()
# plt.plot(path[:,0],path[:,1])


# path = CreatePath(15000,0.8)
# path2maze(path)

# np.savetxt('SimRan15000.csv', path, fmt='%10.5f', delimiter=',')

# path = np.genfromtxt('SimRan15000.csv',delimiter=',')
# plt.figure()
# plt.plot(path[:,0],path[:,1])

# plt.show()
