import numpy as np
import matplotlib.pyplot as plt

def plt_bar(a):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ## the data
    N = int(len(a)/4)
    GG = []
    GP = []
    PP = []
    GGPP = []
    for i in range(0,N):
        GG.append(a[i*4])
        GP.append(a[i*4+1])
        PP.append(a[i*4+2])
        GGPP.append(a[i*4+3])
    # menMeans = [18, 35, 30, 35, 27]
    # Std =   [0.01,0.01,0.01,0.01,0.01,0.01]
    # womenMeans = [25, 32, 34, 20, 25]
    # womenStd =   [3, 5, 2, 3, 3]
    # womenMeans = [25, 32, 34, 20, 25]
    # womenStd =   [3, 5, 2, 3, 3]

    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    ind = ind + 0.1
    width = 0.17                      # the width of the bars

    ## the bars
    rects1 = ax.bar(ind, GG, width,
                    color='red')
    rects2 = ax.bar(ind+width, GP, width,
                        color='green')
    rects3 = ax.bar(ind+width+width, PP, width,
                        color='blue')
    rects4 = ax.bar(ind+width+width+width, GGPP, width,
                        color='yellow')

    # axes and labels
    # ax.set_xlim(-width,len(ind)+width)
    # ax.set_ylim(0,1.1)
    ax.set_ylabel('Similarity Scores')
    ax.set_title('Different Models and Learning Rules')
    xTickMarks = ['Group'+str(i) for i in range(1,7)]
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)

    ## add a legend
    ax.legend( (rects1[0], rects2[0],rects3[0],rects4[0]), ('GG', 'GP','PP','GGPP') )