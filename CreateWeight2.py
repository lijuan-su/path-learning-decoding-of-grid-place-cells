import numpy as np
import matplotlib.pyplot as plt
# import CreateGrid
import CreatePlace
# import CreatePath
from config import *
import ExperimentPath
# import numpy as np
# import matplotlib.pyplot as plt
# import CreateGrid
# import CreatePlace
import PathRandom
# import RetrieveActive
# import ExperimentPath
# import CreateWeight
from config import *
import correlation
import save_figures
import PathReward
import GridLinear
import GridModule
import barplot
import random
import statsmodels.api as sm 


def CreateWeight_modular(path, grids, places):
    GridNum = grids.shape[2]
    PlaceNum = places.shape[2]

    # w_GG = np.zeros([GridNum, GridNum])
    # w_PP = np.zeros([PlaceNum, PlaceNum])
    # w_GP = np.zeros([GridNum, PlaceNum])

    w_GG2 = np.zeros([GridNum, GridNum])
    # w_PP2 = np.zeros([PlaceNum, PlaceNum])
    # w_GP2 = np.zeros([GridNum, PlaceNum])

    alpha = 0.3
    belta = 0.1

    for ii in [2,1,0.5,0.01,0]:
        alpha = 0.3
        belta = 0.3*ii
        for i in range(0,path.shape[0]):
            x = path[i, 0]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              y = path[i, 1]
            for j in range(0, GridNum):
                for k in range(0, GridNum):
                    if (j//(GridNum//4+1) == k//(GridNum//4+1)):
                        alpha1 = alpha
                        belta1 = belta
                    else:
                        alpha1 = alpha * 0.1
                        belta1 = belta * 0.1
                    # w_GG[j,k] = w_GG[j,k] + grids[x,y,j] * grids[x,y,k] * alpha1
                    # w_GP[j,k] = w_GP[j,k] + grids[x,y,j] * places[x,y,k] * alpha
                    # w_PP[j,k] = w_PP[j,k] + places[x,y,j] * places[x,y,k] * alpha

                    if (grids[x,y,j]>0 and grids[x,y,k]>0):
                        w_GG2[j,k] = w_GG2[j,k] + grids[x,y,j] * grids[x,y,k] * alpha1
                    if (grids[x,y,j]>0 and grids[x,y,k]==0):
                        w_GG2[j,k] = w_GG2[j,k] - grids[x,y,j] * belta1
                        w_GG2[k,j] = w_GG2[j,k]

                    # if (grids[x,y,j]>0 and places[x,y,k]>0):
                    #     w_GP2[j,k] = w_GP2[j,k] + grids[x,y,j] * places[x,y,k] * alpha
                    # if (grids[x,y,j]>0 and places[x,y,k]==0):
                    #     w_GP2[j,k] = w_GP2[j,k] - grids[x,y,j] * belta
                    #     w_GP2[k,j] = w_GP2[j,k]

                    # if (places[x,y,j]>0 and places[x,y,k]>0):
                    #     w_PP2[j,k] = w_PP2[j,k] + places[x,y,j] * places[x,y,k] * alpha
                    # if (places[x,y,j]>0 and places[x,y,k]==0):
                    #     w_PP2[j,k] = w_PP2[j,k] - places[x,y,j] * belta
                    #     w_PP2[k,j] = w_PP2[j,k]
            w_GG2 [w_GG2 < 0] = 0
            # w_PP2 [w_PP2 < 0] = 0
            # w_GP2 [w_GP2 < 0] = 0
        # create_weight(grids, places, w_GG)
        create_weight(grids, places, w_GG2)

    # return  w_GG, w_GP, w_PP, w_GG2, w_GP2, w_PP2
    # return  w_GG, w_GG2


def overlap_on_maze(weightMatrix,cells1,cells2):
    sum_maze = np.zeros([MazeSize, MazeSize])
    for i in range(0, GridNum):
            for j in range(0, GridNum):
                sum_maze = sum_maze + weightMatrix[i][j] * cells1[:,:,i]* cells2[:,:,j]
    sum_maze = sum_maze/sum_maze.max()
    return sum_maze

def create_weight(grids, places, w_GG):
    g_sum_maze = overlap_on_maze(w_GG,grids,grids)
    # gp_sum_maze = overlap_on_maze(w_GP,grids,places)
    # p_sum_maze = overlap_on_maze(w_PP,places,places)

    fig = plt.figure()
    ii = 121
    # for i in [w_GG,w_GP,w_PP, g_sum_maze, gp_sum_maze, p_sum_maze]:
    for i in [w_GG, g_sum_maze]:
        plt.subplot(ii)
        cs = plt.imshow(i)
        # fig.colorbar(cs, shrink=0.7, pad=0.02)
        ii += 1


GridNum = 225
PlaceNum = 225
grids = GridModule.CreateGrid(GridNum)
places = CreatePlace.CreatePlace(PlaceNum)
# path = ExperimentPath.path_3rewards_Real()[10000:15000,1:3]
path1 = PathReward.CreatePath3(1500)
CreateWeight_modular(path1, grids, places)

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig('figures/21050010_%s.pdf' % i)
    # # plt.savefig('figures/Sim4RewRan1000_%s.pdf' % i)

plt.show()
