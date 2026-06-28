from array import array

import numpy as np

a = np.array([1, 2, 5, 4, 6])
# print(a.ndim)

# a = np.array([[1, 2, 5, 4 , 6], [2, 3, 4, 5, 6]])
# print(a.ndim)
#ndim => dung de xem co bao nhieu mang

b = np.array([[1, 2, 3, 4, 3], [4, 5, 6, 7, 1]])
# print(b[1][2:4])
# print(b.shape)
# print(a.shape)

#len => dung de dem phan tu mang
# print(len(b[1]))

#arange => random mang
c = np.arange(5)
# print(c)
# print(c[0][0:5])

print(a * b[0])
