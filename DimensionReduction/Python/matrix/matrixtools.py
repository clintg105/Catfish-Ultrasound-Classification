import numpy as np


def partition2D(arr, hsplit, vsplit):
    h, w = arr.shape
    assert h%hsplit==w%vsplit==0
    return np.reshape(arr, (hsplit, vsplit, h//hsplit, w//vsplit)).swapaxes(1,2)
# a = np.arange(100).reshape(10,10)
# print(a, '\n', partition2D(a, 2, 5), '\n', partition2D(a, 2, 5).shape)