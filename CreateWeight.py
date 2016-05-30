import numpy as np
import matplotlib.pyplot as plt
# import CreateGrid
import CreatePlace
# import CreatePath
from config import *
import ExperimentPath

def CreateWeight(path, grids, places):
    GridNum = grids.shape[2]
    PlaceNum = places.shape[2]

    w_GG = np.zeros([GridNum, GridNum])
    w_PP = np.zeros([PlaceNum, PlaceNum])
    w_GP = np.zeros([GridNum, PlaceNum])
    # w_GG_1 = np.zeros([GridNum, GridNum])
    # w_PP_1 = np.zeros([PlaceNum, PlaceNum])
    # w_GP_1 = np.zeros([GridNum, PlaceNum])

    w_GG2 = np.zeros([GridNum, GridNum])
    w_PP2 = np.zeros([PlaceNum, PlaceNum])
    w_GP2 = np.zeros([GridNum, PlaceNum])
    # w_GG_2 = np.zeros([GridNum, GridNum])
    # w_PP_2 = np.zeros([PlaceNum, PlaceNum])
    # w_GP_2 = np.zeros([GridNum, PlaceNum])


    alpha = 0.3
    belta = 0.1

    for i in range(0,path.shape[0]):
        x = path[i, 0]
        y = path[i, 1]
        for j in range(0, GridNum):
            for k in range(0, GridNum):
                w_GG[j,k] = w_GG[j,k] + grids[x,y,j] * grids[x,y,k] * alpha
                w_GP[j,k] = w_GP[j,k] + grids[x,y,j] * places[x,y,k] * alpha
                w_PP[j,k] = w_PP[j,k] + places[x,y,j] * places[x,y,k] * alpha

                if (grids[x,y,j]>0 and grids[x,y,k]>0):
                    w_GG2[j,k] = w_GG2[j,k] + grids[x,y,j] * grids[x,y,k] * alpha
                if (grids[x,y,j]>0 and grids[x,y,k]==0):
                    w_GG2[j,k] = w_GG2[j,k] - grids[x,y,j] * belta
                    w_GG2[k,j] = w_GG2[j,k]

                if (grids[x,y,j]>0 and places[x,y,k]>0):
                    w_GP2[j,k] = w_GP2[j,k] + grids[x,y,j] * places[x,y,k] * alpha
                if (grids[x,y,j]>0 and places[x,y,k]==0):
                    w_GP2[j,k] = w_GP2[j,k] - grids[x,y,j] * belta
                    w_GP2[k,j] = w_GP2[j,k]

                if (places[x,y,j]>0 and places[x,y,k]>0):
                    w_PP2[j,k] = w_PP2[j,k] + places[x,y,j] * places[x,y,k] * alpha
                if (places[x,y,j]>0 and places[x,y,k]==0):
                    w_PP2[j,k] = w_PP2[j,k] - places[x,y,j] * belta
                    w_PP2[k,j] = w_PP2[j,k]
        w_GG2 [w_GG2 < 0] = 0
        w_GP2 [w_GP2 < 0] = 0
        w_PP2 [w_PP2 < 0] = 0

        # if i == int(path.shape[0]/100):
        # if i == 50 :
        #     # print (i)
        #     w_GG_1[:] = w_GG[:]
        #     w_PP_1[:] = w_PP[:]
        #     w_GP_1[:] = w_GP[:]
        #     w_GG_2[:] = w_GG2[:]
        #     w_PP_2[:] = w_PP2[:]
        #     w_GP_2[:] = w_GP2[:]
                
    return  w_GG, w_GP, w_PP, w_GG2, w_GP2, w_PP2

def CreateWeight_modular(path, grids, places):
    GridNum = grids.shape[2]
    PlaceNum = places.shape[2]

    w_GG = np.zeros([GridNum, GridNum])
    w_PP = np.zeros([PlaceNum, PlaceNum])
    w_GP = np.zeros([GridNum, PlaceNum])
    # w_GG_1 = np.zeros([GridNum, GridNum])
    # w_PP_1 = np.zeros([PlaceNum, PlaceNum])
    # w_GP_1 = np.zeros([GridNum, PlaceNum])

    w_GG2 = np.zeros([GridNum, GridNum])
    w_PP2 = np.zeros([PlaceNum, PlaceNum])
    w_GP2 = np.zeros([GridNum, PlaceNum])
    # w_GG_2 = np.zeros([GridNum, GridNum])
    # w_PP_2 = np.zeros([PlaceNum, PlaceNum])
    # w_GP_2 = np.zeros([GridNum, PlaceNum])

    alpha = 0.3
    belta = 0.1

    for i in range(0,path.shape[0]):
        x = path[i, 0]
        y = path[i, 1]
        for j in range(0, GridNum):
            for k in range(0, GridNum):
                if (j//(GridNum//4+1) == k//(GridNum//4+1)):
                    alpha1 = 0.3
                    belta1 = 0.1
                else:
                    alpha1 = 0.03
                    belta1 = 0.01
                w_GG[j,k] = w_GG[j,k] + grids[x,y,j] * grids[x,y,k] * alpha1
                w_GP[j,k] = w_GP[j,k] + grids[x,y,j] * places[x,y,k] * alpha
                w_PP[j,k] = w_PP[j,k] + places[x,y,j] * places[x,y,k] * alpha

                if (grids[x,y,j]>0 and grids[x,y,k]>0):
                    w_GG2[j,k] = w_GG2[j,k] + grids[x,y,j] * grids[x,y,k] * alpha1
                if (grids[x,y,j]>0 and grids[x,y,k]==0):
                    w_GG2[j,k] = w_GG2[j,k] - grids[x,y,j] * belta1
                    w_GG2[k,j] = w_GG2[j,k]

                if (grids[x,y,j]>0 and places[x,y,k]>0):
                    w_GP2[j,k] = w_GP2[j,k] + grids[x,y,j] * places[x,y,k] * alpha
                if (grids[x,y,j]>0 and places[x,y,k]==0):
                    w_GP2[j,k] = w_GP2[j,k] - grids[x,y,j] * belta
                    w_GP2[k,j] = w_GP2[j,k]

                if (places[x,y,j]>0 and places[x,y,k]>0):
                    w_PP2[j,k] = w_PP2[j,k] + places[x,y,j] * places[x,y,k] * alpha
                if (places[x,y,j]>0 and places[x,y,k]==0):
                    w_PP2[j,k] = w_PP2[j,k] - places[x,y,j] * belta
                    w_PP2[k,j] = w_PP2[j,k]
        w_GG2 [w_GG2 < 0] = 0
        w_PP2 [w_PP2 < 0] = 0
        w_GP2 [w_GP2 < 0] = 0

    return  w_GG, w_GP, w_PP, w_GG2, w_GP2, w_PP2


def overlap_on_maze(weightMatrix,cells1,cells2):
    sum_maze = np.zeros([MazeSize, MazeSize])
    for i in range(0, GridNum):
            for j in range(0, GridNum):
                sum_maze = sum_maze + weightMatrix[i][j] * cells1[:,:,i]* cells2[:,:,j]
    sum_maze = sum_maze/sum_maze.max()
    return sum_maze

def firingRate(path, grids, places):
    GridNum = grids.shape[2]
    PlaceNum = places.shape[2]

    firingrate_G = np.zeros(GridNum)
    firingrate_P = np.zeros(PlaceNum)

    step = path.shape[0]
    for i in range(0,step):
        x = path[i, 0]
        y = path[i, 1]
        for j in range(0, GridNum):
            firingrate_G[j] += grids[x,y,j]
            firingrate_P[j] += places[x,y,j]

    return firingrate_G/step, firingrate_P/step
