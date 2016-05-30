import numpy as np
import matplotlib.pyplot as plt
import random
from config import *
import ArrayLocalMax

# Fitting fuction from Experimental Data
def spacing2lamda(spacing):
    # Fitting with some data
    # Norm of residuals = 0.033491
    # Coefficients:
    x = spacing
    p1 = 0.99760841
    p2 = 0.3412051
    p3 = -9.06458861
    y = p1*x*x + p2*x + p3
    # 0.99760841  0.3412051  -9.06458861
    return y


def CreateGrid (GridNum):
    "Using three 2-D Cosine Function Models"
    "Spcing is the field distance of grid cells"
    "Min spacing is 0.2; medium is 0.7; max is 1.2;"
    # spacing = 0.4
    # spacing = 40
    # phasex = int(MazeSize/2)
    # phasey = int(MazeSize/2)
    # phasex = 20
    # phasey = 80
    # orient = 0
    # orient = np.pi/18
    spacing = np.linspace(20,120,GridNum)
    if GridNum ==1:
        spacing[0] = 40
    # spacing_modular = np.array([38,48,65,98])
    last = np.zeros([MazeSize, MazeSize, GridNum])
    random.seed(10)

    for k in range(0, GridNum):

        phasex = random.randint(2, MazeSize-2)
        phasey = random.randint(2, MazeSize-2)
        orient = np.pi/18*random.random()
        # phasex = xx[k]
        # phasey = yy[k]

        x = np.linspace(0,MazeSize-1, MazeSize)
        y = np.linspace(0,MazeSize-1, MazeSize)
        X, Y = np.meshgrid(x, y)
        Xp = X.reshape(len(X)*len(X))
        Yp = Y.reshape(len(Y)*len(Y))

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
        gridfiring = np.exp(0.3*(gridfiring+1.5)) - 1
        M = gridfiring.max()
        gridfiring = gridfiring / float(M)
        
        gridfiring [gridfiring < 0.2] = 0

        # # plt.figure()
        # # plt.imshow(gridfiring)
        # fig = plt.figure()
        # cs = plt.imshow(gridfiring)
        # fig.colorbar(cs, shrink=0.9)
        # local_max = ArrayLocalMax.plot_local_max(gridfiring)
        # plt.scatter(local_max[:,0],local_max[:,1],s=60,alpha=0.5)
        # plt.xlim(0,MazeSize-1)
        # plt.ylim(MazeSize-1,0)
        # print ("({},{})".format(phasex,phasey),orient*180/np.pi,spacing[k])

        last[:, :, k] = gridfiring
    return last

# GridNum = 4 
# grids = CreateGrid(GridNum)

# # # for i in plt.get_figlabels():
# for i in plt.get_fignums():
#     # print ('fig_%s.pdf' % i)
#     plt.figure(i)
#     plt.xlim(0,MazeSize-1)
#     plt.ylim(MazeSize-1,0)
# #     # plt.savefig('figures/G40R10P2080_%s.pdf' % i)
# GridNum = 9 
# grids = CreateGrid(GridNum)
# # plt.figure()
# # plt.imshow(grids[:,:,48])
# fig, ax = plt.subplots()
# im = ax.imshow(grids[:,:,8])
# fig.colorbar(im)
# plt.show()
