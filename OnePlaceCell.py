import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random
from config import *
from scipy.stats import multivariate_normal
import ArrayLocalMax
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# import save_figures
# import ipdb

# Using Gaussian Tuning curves suggested by O'Keefe
def CreatePlace(PlaceNum):
    last = np.zeros([MazeSize, MazeSize, PlaceNum])
    var = np.array([[250.0,0],[0,250]])
    # delta_var = np.array([[500/(PlaceNum-1),0],[0,500/(PlaceNum-1)]])
    # mu = [MazeSize/2.0,MazeSize/2.0]
    # mu = [20, 20]
    mu = [75,75]
    # delta_mu = np.array([MazeSize/PlaceNum, MazeSize/PlaceNum])
    delta_mu = np.array([100/PlaceNum, 100/PlaceNum])
    # mu = [MazeSize/2.0,MazeSize/2.0]
    random.seed(10)

    for k in range(0, PlaceNum):
        # mu = [random.randint(20,MazeSize-20),random.randint(20,MazeSize-20)]
        x, y = np.mgrid[1:MazeSize+1:1, 1:MazeSize+1:1]
        pos = np.empty(x.shape + (2,))
        pos[:,:,0] = x
        pos[:,:,1] = y
        rv = multivariate_normal(mu, var)
        placefiring = rv.pdf(pos)
        # print(placefiring.max(),placefiring.min())

        fig = plt.figure('Gaussian placefiring')
        cs = plt.imshow(placefiring)
        fig.colorbar(cs, shrink=0.9)

        M = placefiring.max()
        placefiring = placefiring / float(M)
        # print(placefiring.max(),placefiring.min())

        # plt.figure()
        # plt.imshow(placefiring)
        fig = plt.figure('Placefiring/Max')
        cs = plt.imshow(placefiring)
        fig.colorbar(cs, shrink=0.9)
        # plt.title("({},{})".format(mu[1],mu[0]),"({},{})".format(var[0,0],var[1][1]),var[0][1])

        # local_max = ArrayLocalMax.plot_local_max(placefiring)
        # # spacing_real.append(local_max[1,1]-local_max[0,1])
        # plt.scatter(local_max[:,0],local_max[:,1],s=60,alpha=0.5)

        fig = plt.figure('surface')
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, placefiring, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        placefiring [placefiring < 0.3] = 0
        fig = plt.figure('Less then 0.3 is zero')
        cs = plt.imshow(placefiring)
        fig.colorbar(cs, shrink=0.9)

        print ("({},{})".format(mu[1],mu[0]),"({},{})".format(var[0,0],var[1][1]),var[0][1])

        # var = var + delta_var
        # mu = mu + delta_mu
        last[:,:,k] = placefiring

    return last

PlaceNum = 1 
places = CreatePlace(PlaceNum)

# for i in [10,20]:
#     plt.figure()
#     plt.imshow(places[:,:,i])

#     fig = plt.figure()
#     cs = plt.contourf(places[:,:,i])
#     fig.colorbar(cs, shrink=0.9)

# for i in plt.get_figlabels():
# # for i in plt.get_fignums():
#     # print ('fig_%s.pdf' % i)
#     plt.figure(i)
#     plt.xlim(0,MazeSize-1)
#     plt.ylim(MazeSize-1,0)
#     plt.savefig('figures/P_Var250_7575_%s.pdf' % i)

plt.show()
