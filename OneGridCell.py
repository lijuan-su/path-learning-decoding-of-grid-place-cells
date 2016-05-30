import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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
    phasex = 20
    phasey = 80
    # orient = 0
    orient = np.pi/18
    spacing = np.linspace(40,120,GridNum)
    last = np.zeros([MazeSize, MazeSize, GridNum])
    # This method failed because of last2 is a reference, which refer to last
    random.seed(10)
    # spacing_real = []

    for k in range(0, GridNum):

        # phasex = random.randint(1, MazeSize-1)
        # phasey = random.randint(1, MazeSize-1)
        # orient = np.pi/6*random.random()

        x = np.linspace(0,MazeSize-1, MazeSize)
        y = np.linspace(0,MazeSize-1, MazeSize)
        X, Y = np.meshgrid(x, y)
        Xp = X.reshape(len(X)*len(X))
        Yp = Y.reshape(len(Y)*len(Y))

        lamda = spacing2lamda(spacing[k])
        theta = np.array([0, np.pi/3, np.pi/3*2]) - orient
        # theta = np.array([0, np.pi/6, np.pi/3]) - orient

        H = [np.cos(theta), np.sin(theta)]
        H = np.array(H)
        H = np.transpose(H)
        xy = np.array([Xp-phasex, Yp-phasey])
        projMesh = np.dot(H,xy)
        
        "Three grating of cosine"
        grating1 = np.cos(projMesh[0,:]*4*np.pi/(np.sqrt(3*lamda))).reshape(np.size(x), -1)
        grating2 = np.cos(projMesh[1,:]*4*np.pi/(np.sqrt(3*lamda))).reshape(np.size(x), -1)
        grating3 = np.cos(projMesh[2,:]*4*np.pi/(np.sqrt(3*lamda))).reshape(np.size(x), -1)

        # gridfiring = grating1 + grating2 + grating3
        # gridfiring = np.exp(0.3*(gridfiring+1.5)) - 1
        # M = gridfiring.max()
        # gridfiring = gridfiring / float(M)

        fig = plt.figure('grating1')
        cs = plt.imshow(grating1)
        fig.colorbar(cs, shrink=0.9)
        # plt.figure()
        # plt.imshow(gridfiring)
        fig = plt.figure('grating2')
        cs = plt.imshow(grating2)
        fig.colorbar(cs, shrink=0.9)
        # plt.figure()
        # plt.imshow(gridfiring)
        fig = plt.figure('grating3')
        cs = plt.imshow(grating3)
        fig.colorbar(cs, shrink=0.9)

        gridfiring = grating1 + grating2 + grating3
        print(gridfiring.max(),gridfiring.min())
        # plt.figure()
        # plt.imshow(gridfiring)
        fig = plt.figure('gratings')
        cs = plt.imshow(gridfiring)
        fig.colorbar(cs, shrink=0.9)

        gridfiring = np.exp(0.3*(gridfiring+1.5)) - 1
        print(gridfiring.max(),gridfiring.min())
        # plt.figure()
        # plt.imshow(gridfiring)
        fig = plt.figure('exp(0.3*(gridfiring+1.5))-1')
        cs = plt.imshow(gridfiring)
        fig.colorbar(cs, shrink=0.9)

        M = gridfiring.max()
        gridfiring = gridfiring / float(M)
        print(gridfiring.max(),gridfiring.min())

        fig = plt.figure('gridfiring_div_max')
        cs = plt.imshow(gridfiring)
        fig.colorbar(cs, shrink=0.9)

        # local_max = ArrayLocalMax.plot_local_max(gridfiring)
        # # spacing_real.append(local_max[1,1]-local_max[0,1])
        # plt.scatter(local_max[:,0],local_max[:,1],s=60,alpha=0.5)

        fig = plt.figure('gridfiring_div_max_contourf')
        cs = plt.contourf(X,Y,gridfiring)
        fig.colorbar(cs, shrink=0.9)
        # plt.scatter(local_max[:,1],local_max[:,0],s=np.pi*6*6,alpha=0.5)

        fig = plt.figure('surface')
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, gridfiring, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        gridfiring [gridfiring < 0.3] = 0
        fig = plt.figure('Less then 0.3 is zero')
        cs = plt.imshow(gridfiring)
        fig.colorbar(cs, shrink=0.9)

        last[:, :, k] = gridfiring
        # spacing = spacing + delta_spaceing
        # phasex = phasex + delta_phase
        # phasey = phasey + delta_phase

        # print (spacing[k],phasex,phasey,orient*180/np.pi)
        print ("({},{})".format(phasex,phasey),orient*180/np.pi,spacing[k])
    # plt.figure()
    # plt.plot(spacing,spacing_real)
    # poly_fit(spacing,spacing_real,2)
    # poly_fit(spacing,spacing_real,3)
    # poly_fit(spacing,spacing_real,4)
    return last


def poly_fit(x,y,degree):
    # # spacing = np.linspace(20,120,GridNum)
    # # spacing_real = []
    # # spacing_real.append(local_max[1,1]-local_max[0,1])
    # # plt.figure()
    # # plt.plot(spacing,spacing_real)
    # # poly_fit(spacing,spacing_real,2)
    # # poly_fit(spacing,spacing_real,3)
    # # poly_fit(spacing,spacing_real,4)
    z = np.polyfit(x,y,degree)
    print (z)
    p = np.poly1d(z)
    xp = np.linspace(20,120,GridNum)
    plt.figure()
    plt.plot(x,y,'.',xp,p(xp),'-')
    yfit = p(x)
    res = np.sqrt(sum((yfit-y)**2))/GridNum
    print (res)

GridNum = 1 
grids = CreateGrid(GridNum)

# for i in plt.get_figlabels():
# # for i in plt.get_fignums():
#     # print ('fig_%s.pdf' % i)
#     plt.figure(i)
#     plt.xlim(0,MazeSize-1)
#     plt.ylim(MazeSize-1,0)
#     plt.savefig('figures/G40R10P2080_%s.pdf' % i)
plt.show()



