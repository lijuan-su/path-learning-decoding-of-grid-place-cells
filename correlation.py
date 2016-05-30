import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import random
# import CreateGrid
import CreatePlace
from config import *
# from skimage.measure import structural_similarity as ssim
# from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as patches
import save_figures
import GridLinear
import GridModule

def corrcoef_cells(cell1, cell2):
    cor12 = np.corrcoef(cell1.reshape(-1), cell2.reshape(-1))[0][1]
    return cor12 

def mse(cell1, cell2):
    err = np.sum((cell1.reshape(-1) - cell2.reshape(-1))**2)
    err /= cell1.shape[1]
    return err

def multi(cell1, cell2):
    multi_lap = cell1 * cell2
    multi_sum = multi_lap.sum()
    return multi_sum

def overlaps(cells1,cells2):
    k1 = cells1.shape[2]
    k2 = cells2.shape[2]
    # print (k1,k2)
    corr_GG = np.zeros((k1,k2))
    err = np.zeros((k1,k2))
    multi_over = np.zeros((k1,k2))
    for i in range(0,k1):
        for j in range(0,k2):
            corr_GG[i,j] = corrcoef_cells(cells1[:,:,i], cells2[:,:,j])
            err[i,j] = mse(cells1[:,:,i], cells2[:,:,j])
            multi_over[i,j] = multi(cells1[:,:,i], cells2[:,:,j])

    err = 1 - err / err.max()
    corr_GG = np.abs(corr_GG)
    mul = multi_over/multi_over.max()
    return corr_GG, err, mul

def cross_correlation(cells1,cells2):
    k1 = cells1.shape[2]
    k2 = cells2.shape[2]
    # print (k1,k2)
    # corr_GG = np.zeros((k1,k2))
    multi_over = np.zeros((k1,k2))
    mul2 = np.zeros((k1,k2))
    for i in range(0,k1):
        for j in range(0,k2):
            # corr_GG[i,j] = corrcoef_cells(cells1[:,:,i], cells2[:,:,j])
            # err[i,j] = mse(cells1[:,:,i], cells2[:,:,j])
            multi_over[i,j] = multi(cells1[:,:,i], cells2[:,:,j])
            mul2 [i,j] = np.correlate(cells1[:,:,i].reshape(-1), cells2[:,:,j].reshape(-1))

    # err = 1 - err / err.max()
    # corr_GG = np.abs(corr_GG)
    mul = multi_over/multi_over.max()
    return mul,mul2


def plot_overlap_twocells(cells,i,j):
    plt.figure()
    plt.subplot(221)
    plt.imshow(cells[:,:,i])  # The AxesPlace object work as a list of axes.
    plt.subplot(222)
    plt.imshow(cells[:,:,j])  # The AxesPlace object work as a list of axes.
    plt.subplot(223)
    overlap_cells(cells,i,j)
    plt.subplot(224)
    plt.imshow(cells[:,:,i]*cells[:,:,j])  # The AxesPlace object work as a list of axes.
    # plt.subplot(224)
    # plt.imshow(cells[:,:,i]*cells[:,:,j]*9)  # The AxesPlace object work as a list of axes.

def overlap_cells(cells,i,j):
    extent = 0,150,150,0
    Z1 = cells[:,:,i]
    Z2 = cells[:,:,j]
    im1 = plt.imshow(Z1, cmap=plt.cm.gray, alpha=.7, interpolation='bilinear',extent=extent)
    plt.hold(True)
    im2 = plt.imshow(Z2, cmap=plt.cm.jet, alpha=.3, interpolation='bilinear',extent=extent)

# GridNum = 16
# grids1 = GridLinear.CreateGrid(1)
# grids2 = GridLinear.CreateGrid(400)
# cor1,cor2 = cross_correlation(grids1,grids2)
# plt.figure()
# x = np.linspace(20,120,400)
# plt.plot(x,cor1[0,:])
# plt.figure()
# plt.plot(x,cor2[0,:])
# # places = CreatePlace.CreatePlace(PlaceNum)

# # p_corr,p_err,p_mul = overlaps(places,places)
# g_corr,g_err,g_mul = overlaps(grids,grids)
# plot_overlap_twocells(grids,0,1)
# plot_overlap_twocells(grids,1,2)

# # for i in [p_corr,p_err,p_mul,g_corr,g_err,g_mul]:
# for i in [g_corr,g_err,g_mul]:
#     fig, ax = plt.subplots()
#     im = ax.imshow(i)
#     fig.colorbar(im)

# plt.show()
# save_figures.multipage('correlation.pdf')
# # multipage('multipage_w_raster.pdf')
