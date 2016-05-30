import numpy as np
import matplotlib.pyplot as plt
from config import *
import math
# import ipdb
import save_figures

#  082714.txt contains the Random Path from the Experiment Data
#  Sleep: 2569:34150 
#  Task: 38827:71490 176716:206193

#  081412.txt contains the Path With 3 Rewards from the Experiment Data
#  Task: 10641:35073

#  data11=load('090614.txt')   Task: 32662:63191 104158:134366 174172:204328
#  data12=load('082914.txt')   Task: 34974:64219 109953:139371 180193:209970
#  data13=load('082714.txt')   Sleep:2569:34150 Task: 38827:71490 176716:206193
#  data21=load('081412.txt')   Task: 10641:35073
#  data22=load('073114.txt')
#  data23=load('072614.txt')

## Extract array[time, x, y] data from .txt 
def path_extract_txt(dataname,start_line,end_line):
    path_list = []
    # with open('081412.txt','r') as f:
    with open(dataname,'r') as f:
        fp = f.readlines()
        # print (len(fp), type(fp),fp[0:2],type(fp[0]))
        # >>> 239600 <class 'list'> ['12323873377\t331\t228\n', '12323914082\t331\t228\n'] <class 'str'>  
        temp = fp[start_line : end_line]
        # print (len(temp), temp)
        for element in temp:
            # print (type(element))
            temp1 = element.strip('\n')
            # print (type(temp1),temp1)
            temp2 = temp1.split()
            # print (type(temp2),type(temp2[0]),len(temp2),temp2)
            temp3 = [int(i) for i in temp2]
            # print (type(temp3),type(temp3[0]),len(temp3),temp3)
            path_list.append(temp3)
    # print('Finish')
    path_random = np.array(path_list)
    return path_random

def path_random_Real():
    path_random_pixel= path_extract_txt('082714.txt',176716,196716)
    # path_random_pixel= path_extract_txt('082714.txt',176716,176816)
    path_random1 = path_random_pixel 
    path_random1[:,1] = path_random_pixel[:,1]/3-35
    path_random1[:,2] = path_random_pixel[:,2]/3-13
    path_random = np.array([item for item in path_random1 if np.sqrt((item[1]-MazeSize/2)**2+(item[2]-MazeSize/2)**2) < MazeSize/2.0])
    return path_random

def path_3rewards_Real():
    path_3rewards_pixel= path_extract_txt('081412.txt',15073,35073)
    path_3rewards1 = path_3rewards_pixel 
    path_3rewards1[:,1] = path_3rewards_pixel[:,1]/3-32
    path_3rewards1[:,2] = path_3rewards_pixel[:,2]/3
    path_3rewards = np.array([item for item in path_3rewards1 if np.sqrt((item[1]-MazeSize/2)**2+(item[2]-MazeSize/2)**2) < MazeSize/2.0])
    return path_3rewards

def plot_path(path,circle=True):
    plt.plot(path[:,1],path[:,2])
    if circle == True :
        circle = plt.Circle((MazeSize/2.0,MazeSize/2.0),MazeSize/2.0, color ='r', fill = False)
        fig = plt.gcf()
        fig.gca().add_artist(circle)
        plt.axis([0,150,0,150])
    plt.axes().set_aspect('equal')


def subplot_path(path_random, path_3rewards):
    # path_random = path_random_Real()
    # path_3rewards = path_3rewards_Real()
    plt.figure()
    plt.title('Path of Random and 3rewards')
    sub1 = plt.subplot(121)
    plt.plot(path_random[:,1],path_random[:,2])
    circle = plt.Circle((MazeSize/2.0,MazeSize/2.0),MazeSize/2.0, color ='r', fill = False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    sub1.set_xlim([0,150])
    sub1.set_ylim([150,0])
    sub1.set_aspect('equal')

    sub2 = plt.subplot(122)
    plt.plot(path_3rewards[:,1],path_3rewards[:,2])
    circle = plt.Circle((MazeSize/2.0,MazeSize/2.0),MazeSize/2.0, color ='r', fill = False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    sub2.set_xlim([0,150])
    sub2.set_ylim([150,0])
    sub2.set_aspect('equal')

def path_speed(path):
    # path = path_3rewards_Real()
    t = path.shape[0]
    # speed = np.zeros(())
    # speed = [ np.sqrt((path[i+1,1]-path[i,1])**2+(path[i+1,2]-path[i,2])**2)/(path[i+1,0]-path[i,0]) for i in range(0,t-1)]
    speed = [ np.sqrt((path[i+1,1]-path[i,1])**2+(path[i+1,2]-path[i,2])**2) for i in range(0,t-1)]
    # speed.append(speed[-1])
    # speed = np.array(speed)
    # speed = (speed+1)/(speed.max()+1)
    # plt.figure();
    # plt.plot(speed)
    return speed

def path_speed_ExpRew():
    path = path_3rewards_Real()[0000:15000,:]
    t = path.shape[0]
    # speed = np.zeros(())
    # speed = [ np.sqrt((path[i+1,1]-path[i,1])**2+(path[i+1,2]-path[i,2])**2)/(path[i+1,0]-path[i,0]) for i in range(0,t-1)]
    speed = [ np.sqrt((path[i+1,1]-path[i,1])**2+(path[i+1,2]-path[i,2])**2) for i in range(0,t-1)]
    speed.append(speed[-1])
    speed = np.array(speed)
    speed = (speed+1)/(speed.max()+1)
    speed[speed>0.5] = 0.5
    speed = speed/speed.max()
    # plt.figure();
    # plt.plot(speed)
    return speed

def PlotCellPath(path, cells, N):
    # actives = RetrieveActive(path, cells)
    plt.figure()
    plt.plot(path[:,1],path[:,2])
    plt.axis([0,150,0,150])
    t = path.shape[0]
    firing_threhold = 0.3 * cells[:,:,N].max()
    for i in range(0,t):
        x = path[i,1]
        y = path[i,2]
        if (cells[x,y,N] > firing_threhold):
            plt.scatter(x,y,color='red')



def inter_equal_distance(path):
    x = path[:,1]
    y = path[:,2]
    # M = 1000
    # t = np.linspace(0, len(x), M)
    # x = np.interp(t, np.arange(len(x)), x)
    # y = np.interp(t, np.arange(len(y)), y)
    tol = 5
    # i is the length of the path; idx is the extract index; dis is the sum of distace between to idx;
    i, idx,dis = 0, [0], []
    while i < len(x)-1:
        # print (i)
        total_dist = 0
        for j in range(i+1, len(x)):
            # ipdb.set_trace()
            total_dist += math.sqrt((x[j]-x[i])**2 + (y[j]-y[i])**2)
            if total_dist >= tol:
                idx.append(j)
                # dis.append(total_dist/(path[j,0]-path[i,0]))
                dis.append(total_dist)
                i = j
                break
        i = i+1

    xn = x[idx]
    yn = y[idx]
    path_inter = path[idx]

    return path_inter

#  data11=load('090614.txt')   Task: 32662:63191 104158:134366 174172:204328 random!
#  data12=load('082914.txt')   Task: 34974:64219 109953:139371 180193:209970 random!
#  data13=load('082714.txt')   Sleep:2569:34150 Task: 38827:71490 176716:206193 random!
#  data21=load('081412.txt')   Task: 10641,35073
#  data22=load('073114.txt')
#  data23=load('072614.txt')   Task: 47795,67142; 

# PlotCellPath(path, cells, N)
# path1 = path_random_Real()
# path2 = path_3rewards_Real()
# s1 = np.array(path_speed(path2))
# print (s1.shape)
# print (s1.max())
# print (s1.min())
# path1 = path_random_Real()[:,1:3]
# path2 = path_3rewards_Real()[:,1:3]
# plt.figure();
# plt.plot(path1[:,0],path1[:,1])
# plt.figure();
# plt.plot(path2[:,0],path2[:,1])
# path_speed_ExpRew()
# plt.show()
