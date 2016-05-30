import numpy as np
import matplotlib.pyplot as plt
# import CreateGrid
import CreatePlace
import PathRandom
# import RetrieveActive
import ExperimentPath
import CreateWeight
from config import *
import correlation
import save_figures
import PathReward
import GridLinear
import GridModule
import barplot
import random
import statsmodels.api as sm

def plot_path(path):
    plt.figure()
    plt.plot(path[:,0],path[:,1])
    # plt.xlim([0,150])
    # plt.ylim([150,0])
    # ExperimentPath.subplot_path(path, path2)
    # ExperimentPath.PlotCellPath(path, grids, int(GridNum/2))

def path2maze(path):
    path_maze = np.zeros([MazeSize, MazeSize])
    for i in range(0,path.shape[0]):
        x = path[i, 0]
        y = path[i, 1]
        path_maze[x-1,y-1] += 1
    path_maze = path_maze/path_maze.max()

    fig = plt.figure()
    plt.subplot(121)
    # cs = plt.imshow(path)
    plt.plot(path[:,0],path[:,1])
    plt.xlim([0,150])
    plt.ylim([150,0])
    # plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('path')
    plt.subplot(122)
    cs = plt.imshow(path_maze)
    plt.title('path_maze')
    # fig.colorbar(cs, shrink=0.7, pad=0.02)
    return path_maze

def overlaps(grids,places):
    # correlation.plot_overlap_twocells(grids,0,1)
    # correlation.plot_overlap_twocells(places,0,1)

    g_corr,g_err,g_mul = correlation.overlaps(grids,grids)
    p_corr,p_err,p_mul = correlation.overlaps(places,places)
    gp_corr,gp_err,gp_mul = correlation.overlaps(grids,grids)

    fig = plt.figure()
    plt.title('Overlap between Cells')
    ii = 331
    for i in [g_corr,g_err,g_mul,p_corr,p_err,p_mul,gp_corr,gp_err,gp_mul]:
        plt.subplot(ii)
        cs = plt.imshow(i)
        # fig.colorbar(cs, shrink=0.9, pad=0.02)
        ii += 1

def create_weight(path,grids,places,flag):
    if flag == 1:
        weights = CreateWeight.CreateWeight_modular(path,grids,places)
    else:
        weights = CreateWeight.CreateWeight(path,grids,places)
    for j in range(0,int(len(weights)/3)):
        w_GG = weights[j*3]
        w_GP = weights[j*3+1]
        w_PP = weights[j*3+2]

        g_sum_maze = CreateWeight.overlap_on_maze(w_GG,grids,grids)
        gp_sum_maze = CreateWeight.overlap_on_maze(w_GP,grids,places)
        p_sum_maze = CreateWeight.overlap_on_maze(w_PP,places,places)

        for i in [g_sum_maze, gp_sum_maze, p_sum_maze]:
            similar.append(correlation.corrcoef_cells(path2_maze,i))
            similar.append(correlation.mse(path2_maze,i))
            similar.append(correlation.multi(path2_maze,i))
            # n += 1

        fig = plt.figure()
        ii = 231
        for i in [w_GG,w_GP,w_PP, g_sum_maze, gp_sum_maze, p_sum_maze]:
            plt.subplot(ii)
            cs = plt.imshow(i)
            # fig.colorbar(cs, shrink=0.7, pad=0.02)
            ii += 1

        w1, w2, w3 = g_sum_maze, p_sum_maze, gp_sum_maze 
        fit, yy = reg_m(path2_maze, w1, w2,w3)
        print (correlation.corrcoef_cells(path2_maze,yy))
        print (correlation.mse(path2_maze,yy))
        print (correlation.multi(path2_maze,yy))
        fig = plt.figure()
        plt.subplot(121)
        cs = plt.imshow(path2_maze)
        plt.subplot(122)
        cs = plt.imshow(yy)
        
    return g_sum_maze, p_sum_maze, gp_sum_maze, yy

def firingrate(path,g,p,name):
    fg,fp = CreateWeight.firingRate(path,g,p)
    fig = plt.figure()
    plt.subplot(211)
    # calc the trendline (it is simply a linear fitting)
    x = np.arange(GridNum)
    z = np.polyfit(x, fg, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),'r-')
    plt.plot(x,fg)
    plt.title('GridCells Firing Rate on '+name)

    plt.subplot(212)
    x = np.arange(GridNum)
    z = np.polyfit(x, fp, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),'r-')
    plt.plot(x,fp)
    plt.title('PlaceCells Firing Rate on '+name)

def reg_m(y, x1,x2,x3):
    y = y.reshape(-1)
    x = np.array([x1.reshape(-1),x2.reshape(-1), x3.reshape(-1)])
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    yy = results.params[0]*x1 + results.params[1]*x2 + results.params[2]*x3 + results.params[3] 
    yy = yy/yy.max()
    print (results.params)
    print (results.summary())
    return results,yy

# random.seed(10)
grids1 = GridLinear.CreateGrid(GridNum)
grids2 = GridModule.CreateGrid(GridNum)
places = CreatePlace.CreatePlace(PlaceNum)
path0 = PathRandom.CreatePath(5000,0.5)
# path1 = PathReward.CreatePath3(1500)
# path2 = PathReward.CreatePath4(1000)
# path2_maze = path2maze(path2)
# np.random.shuffle(path2)
# plot_path(path0)
# plot_path(path1)
# plot_path(path2)
# path3 = ExperimentPath.path_random_Real()[:5000:,1:3]
# path4 = ExperimentPath.path_3rewards_Real()[0:15000,1:3]
def plot_cell_path():
    path3 = ExperimentPath.path_random_Real()
    path4 = ExperimentPath.path_3rewards_Real()
    ExperimentPath.PlotCellPath(path3, grids1, int(GridNum/3))
    ExperimentPath.PlotCellPath(path3, places, int(GridNum/3))
    ExperimentPath.PlotCellPath(path4, grids1, int(GridNum/3))
    ExperimentPath.PlotCellPath(path4, places, int(GridNum/3))

# plot_cell_path()

similar = []
# for path in [path0,path1,path2]:
for path in [path0]:
    path2_maze = path2maze(path)
    np.random.shuffle(path)
    overlaps(grids1,places)
    overlaps(grids2,places)
    create_weight(path,grids1,places,flag=0)
    create_weight(path,grids2,places,flag=0)
    create_weight(path,grids2,places,flag=1)
    # firingrate(path,grids1,places,'random path')
print (similar)
# np.savetxt('SimRan5000.csv', similar, fmt='%10.5f', delimiter=',')
# np.savetxt('Sim4RewRan1000.csv', similar, fmt='%10.5f', delimiter=',')
sim = np.array(similar).reshape((-1,3))
barplot.plt_bar(sim[:,0]/sim[:,0].max())
barplot.plt_bar(1-sim[:,1]/sim[:,1].max())
barplot.plt_bar(sim[:,2]/sim[:,2].max())

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig('figures/SimRan5000_225_%s.pdf' % i)
    # # plt.savefig('figures/Sim4RewRan1000_%s.pdf' % i)

plt.show()

