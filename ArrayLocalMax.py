import numpy as np
import matplotlib.pyplot as plt
# import ipdb
def plot_local_max(arr):
    local_max = []
    local_range = 10
    n_row,n_col = arr.shape

    for i in range(0,n_row):
        for j in range(0,n_col):
            # ipdb.set_trace()
            Left = max(j-local_range,0)
            Right = min(j+local_range,n_col)
            Up = max(i-local_range,0)
            Down = min(i+local_range,n_row)

            ArrTemp = arr[Up:Down,Left:Right]
            if arr[i,j] == ArrTemp.max() and i != 0 and j != 0 and i != n_row-1 and j != n_col-1:
                # print (i,j)
                # print (Up,Down,Left,Right)
                # temp = np.array([i,j]) + np.array([Up,Left]) 
                temp = np.array([j,i]) 
                # print (temp)
                local_max.append(temp)

    temp1 = list(set(tuple(p) for p in local_max))
    temp1.sort()
    temp2 = np.array(temp1)
    return temp2

# a = np.random.random((6,6))
# print (a)
# b=plot_local_max(a)
# print (b)
# print (np.unique(b))
# c = np.array(list(set(tuple(p) for p in b)))
# print (c)
