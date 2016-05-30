import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from config import *
from scipy.stats import multivariate_normal
import ArrayLocalMax
# import save_figures
# import ipdb

# Using Gaussian Tuning curves suggested by O'Keefe
def CreatePlace(PlaceNum):
    last = np.zeros([MazeSize, MazeSize, PlaceNum])
    gridfield = np.linspace(200,650,PlaceNum)
    # var = np.array([[400.0,0],[0,400]])
    # mu = [75,75]
    # delta_mu = np.array([100/PlaceNum, 100/PlaceNum])
    random.seed(10)

    num = np.sqrt(PlaceNum)
    phase = np.linspace(1,MazeSize-1,num)
    phase = np.linspace(1,MazeSize-1,num)
    XX,YY = np.meshgrid(phase,phase)
    xx = XX.reshape(len(XX)*len(XX))
    yy = YY.reshape(len(YY)*len(YY))
    

    for k in range(0, PlaceNum):
        mu = [random.randint(2,MazeSize-2),random.randint(2,MazeSize-2)]
        # mu = [xx[k],yy[k]]
        var = np.array([[gridfield[k],0],[0,gridfield[k]]])

        x, y = np.mgrid[1:MazeSize+1:1, 1:MazeSize+1:1]
        pos = np.empty(x.shape + (2,))
        pos[:,:,0] = x
        pos[:,:,1] = y
        rv = multivariate_normal(mu, var)
        placefiring = rv.pdf(pos)

        M = placefiring.max()
        placefiring = placefiring / float(M)
        placefiring [placefiring < 0.2] = 0

        # # plt.figure()
        # # plt.imshow(placefiring)
        # fig = plt.figure()
        # cs = plt.imshow(placefiring)
        # fig.colorbar(cs, shrink=0.9)
        # # local_max = ArrayLocalMax.plot_local_max(placefiring)
        # # plt.scatter(local_max[:,0],local_max[:,1],s=60,alpha=0.5)
        # plt.xlim(0,MazeSize-1)
        # plt.ylim(MazeSize-1,0)
        # print ("({},{})".format(mu[1],mu[0]),"({},{})".format(var[0,0],var[1][1]),var[0][1])

        last[:,:,k] = placefiring

    return last

# PlaceNum = 500
# places = CreatePlace(PlaceNum)
# np.savetxt('Place500.csv', places, fmt='%10.5f', delimiter=',')

# # plt.figure()
# # plt.imshow(places[:,:,48])
# fig, ax = plt.subplots()
# im = ax.imshow(places[:,:,2])
# fig.colorbar(im)


# for i in plt.get_fignums():
# # for i in plt.get_figlabels():
#     plt.figure(i)
# #     plt.xlim(0,MazeSize-1)
# #     plt.ylim(MazeSize-1,0)

# for i in plt.get_figlabels():
#     plt.figure(i)
#     plt.savefig('figures/P_Var250_7575_%s.pdf' % i)

# plt.show()

