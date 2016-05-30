import numpy as np
import matplotlib.pyplot as plt

def plt_bar(a):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ## the data
    N = int(len(a)/3)
    GG = []
    GP = []
    PP = []
    for i in range(0,N):
        GG.append(a[i*3])
        GP.append(a[i*3+1])
        PP.append(a[i*3+2])
    # menMeans = [18, 35, 30, 35, 27]
    # Std =   [0.01,0.01,0.01,0.01,0.01,0.01]
    # womenMeans = [25, 32, 34, 20, 25]
    # womenStd =   [3, 5, 2, 3, 3]
    # womenMeans = [25, 32, 34, 20, 25]
    # womenStd =   [3, 5, 2, 3, 3]

    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    ind = ind + 0.17
    width = 0.22                      # the width of the bars

    ## the bars
    rects1 = ax.bar(ind, GG, width,
                    color='red')
    rects2 = ax.bar(ind+width, GP, width,
                        color='green')
    rects3 = ax.bar(ind+width+width, PP, width,
                        color='blue')

    # axes and labels
    # ax.set_xlim(-width,len(ind)+width)
    # ax.set_ylim(0,1.1)
    ax.set_ylabel('Similarity Scores')
    # ax.set_title('Different Models and Learning Rules')
    xTickMarks = ['Group'+str(i) for i in range(1,7)]
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)

    ## add a legend
    ax.legend( (rects1[0], rects2[0],rects3[0]), ('GG', 'GP','PP') )

# a = np.array([[  0.11700293,   6.15026112,   9.26385483],
#        [  0.09416932,   5.28821919,   8.54657383],
#        [  0.12167012,   3.8804992 ,   9.3279166 ],
#        [  0.12036247,   5.01857207,   8.93574177],
#        [  0.09750537,   5.14891287,   8.72759513],
#        [  0.12726231,   3.56504099,   9.3535467 ],
#        [  0.11842032,   7.10428573,   9.01460376],
#        [  0.09416932,   5.28821919,   8.54657383],
#        [  0.12865315,   3.49536425,   9.27549646],
#        [  0.12518787,   6.26507818,   9.30714383],
#        [  0.09750537,   5.14891287,   8.72759513],
#        [  0.12341232,   3.77898383,   9.27050413],
#        [  0.09654635,  11.06004   ,   9.10050269],
#        [  0.09416932,   5.28821919,   8.54657383],
#        [  0.12865315,   3.49536425,   9.27549646],
#        [  0.09957932,  10.62614582,   9.46296556],
#        [  0.09750537,   5.14891287,   8.72759513],
#        [  0.12341232,   3.77898383,   9.27050413]])
# plt_bar(a[:,0]/a[:,0].max())
# plt_bar(a[:,1]/a[:,1].max())
# plt_bar(a[:,2]/a[:,2].max())
# plt.show()
