from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# import matplotlib.pyplot as plt
# import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import random
from config import *
import ArrayLocalMax
from scipy.signal import argrelextrema

# Fitting fuction from Experimental Data
def spacing2lamda(spacing):
    # Fitting with some data
    # Norm of residuals = 0.033491
    # Coefficients:
    x = spacing
    p1 = 0.99760841
    p2 = 0.3420051
    p3 = -9.06458861
    y = p1*x*x + p2*x + p3
    # 0.99760841  0.3420051  -9.06458861
    return y


def CreateGrid (GridNum,lam, phase, ori):
    "Using three 2-D Cosine Function Models"
    "Spcing is the field distance of grid cells"
    "Min spacing is 0.2; medium is 0.7; max is 1.2;"
    # phasex = int(MazeSize/2)
    # phasey = int(MazeSize/2)
    phasex = phase
    phasey = phase
    orient = ori
    # orient = np.pi/18
    spacing = np.linspace(20,200,GridNum)
    if GridNum == 1:
        spacing[0] = lam
    # spacing_modular = np.array([38,48,65,98])
    last = np.zeros([MazeSize,MazeSize, GridNum])
    random.seed(10)

    for k in range(0, GridNum):

        # phasex = random.randint(2, MazeSize-2)
        # phasey = random.randint(2, MazeSize-2)
        # orient = np.pi/18*random.random()

        x = np.linspace(0,MazeSize-1, MazeSize)
        y = np.linspace(0,MazeSize-1, MazeSize)
        # y = np.linspace(0,MazeSize-1, 1)
        # y = np.linspace(0,0,1)
        # print (x.shape,y.shape)
        X, Y = np.meshgrid(x, y)
        # print (X.shape,Y.shape,len(X),len(Y))
        # Xp = X.reshape(len(X)*len(Y))
        # Yp = Y.reshape(len(X)*len(Y))
        Xp = X.reshape((-1))
        Yp = Y.reshape((-1))
        # print (Xp.shape,Yp.shape)

        # lamda = spacing2lamda(spacing_modular[k//(GridNum//4+1)])
        # lamda = spacing2lamda(np.random.normal(spacing_modular[k//(GridNum//4+1)],0.1))
        lamda = spacing2lamda(spacing[k])
        theta = np.array([0, np.pi/3, np.pi/3*2]) - orient

        H = [np.cos(theta), np.sin(theta)]
        H = np.array(H)
        H = np.transpose(H)
        xy = np.array([Xp-phasex, Yp-phasey])
        projMesh = np.dot(H,xy)
        
        "Three grating of cosine"
        grating1 = np.cos(projMesh[0,:]*4*np.pi/(np.sqrt(3*lamda))).reshape(np.size(x), -1)
        grating2 = np.cos(projMesh[1,:]*4*np.pi/(np.sqrt(3*lamda))).reshape(np.size(x), -1)
        grating3 = np.cos(projMesh[2,:]*4*np.pi/(np.sqrt(3*lamda))).reshape(np.size(x), -1)

        gridfiring = grating1 + grating2 + grating3
        # gridfiring = grating1
        gridfiring = np.exp(0.3*(gridfiring+1.5)) - 1
        M = gridfiring.max()
        gridfiring = gridfiring / float(M)

        # phasex = random.randint(2, MazeSize-2)
        # phasey = random.randint(2, MazeSize-2)
        
        last[:, :, k] = gridfiring
    return last

def cross_correlation(cells1,cells2):
    k1 = cells1.shape[2]
    k2 = cells2.shape[2]
    mul2 = np.zeros((k1,k2))
    for i in range(0,k1):
        for j in range(0,k2):
            mul2 [i,j] = np.correlate(cells1[:,:,i].reshape(-1), cells2[:,:,j].reshape(-1))
    mul2 = mul2/mul2.max()
    return mul2

MazeSize = 100
GridNum = 100
max_lamda = 200
x = np.linspace(20,max_lamda,GridNum)
# for i in [40, 60, 80,100]:
corr = []
for i in [40]:
    for phase in range(20,180,50):
    # for phase in range(0,300):
        grids1 = CreateGrid(1,i,phase, 0)
        grids = CreateGrid(GridNum,0,phase, 0)
        corr.append(cross_correlation(grids1,grids).reshape(-1))
    cor = np.average(np.array(corr),axis=0)
    corr = np.array(corr)
    local_max_index = argrelextrema(cor.reshape(-1),np.greater)[0]
    print (corr.shape)
    # plt.plot(x,cor[0,:])
    # plt.plot(x[local_max_index], cor[0,local_max_index], "o", label="min")
    fig = plt.figure()
    # plt.imshow(corr)
    ax = fig.gca(projection='3d')
    print (x.shape)
    y = np.array(range(20,180,50))
    print (y.shape)
    print (corr.shape)
    ax.plot(y,x, corr, label='parametric curve')
    
    # plt.xlabel('lamda (cm)')
    # plt.ylabel('Normalized Cross-Correlation')

    # diff = [(x[local_max_index][i])/x[local_max_index][i-1] for i in range(1, len(local_max_index))]
    # ave = [np.average(diff) for i in range(20, max_lamda)]
    # print (np.average(diff))
    # plt.figure()
    # plt.scatter(x[local_max_index][1:],diff)
    # plt.plot(range(20, max_lamda),ave, color='r', linewidth=1.5)
    # plt.xlabel('Peak Location (cm)')
    # plt.xlim(20,200)
    # plt.ylabel('Peak Ratio')

plt.show()



