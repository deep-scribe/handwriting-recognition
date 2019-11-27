import numpy as np

def dtw(s, t):
    '''
    Calculate the Dynamic Time Warping distance of two time series

    Parameters:
        s: a time series data of dimension (n, ...)
        t: a time series data of dimension (m, ...)

    Return:
        dtw(float): a scalar distance between s and t
    '''
    n = s.shape[0]
    m = t.shape[0]
    # print(n ,m)
    dtw = np.zeros((n,m))
    for i in range(1, n):
        for j in range(1, m):
            dist = np.linalg.norm(s[i]-t[j])
            # print(dist)
            dtw[i,j] = dist + np.min([dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1]])
    return dtw[n-1,m-1]

def l2(s, t):
    '''
    Calculate the l2 distance of two time series

    Parameters:
        s: a vector of dimension (l, )
        t: a vector of dimension (l, )

    Return:
        l2(float): a scalar distance between s and t
    '''
    return np.linalg.norm(s-t)
