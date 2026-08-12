import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from display_network import *

mndata = MNIST('./data')
mndata.load_testing()
X = np.array(mndata.test_images)
cluster_id = 0
K = 10

kmeans = KMeans(n_clusters=K, random_state=11).fit(X)
pred = kmeans.predict(X)

print(type(kmeans.cluster_centers_.T))
print(kmeans.cluster_centers_.T.shape)

# /*** old 
# A = display_network(kmeans.cluster_centers_.T, K, 1)

# f1 = plt.imshow(A, interpolation='nearest', cmap='jet')
# f1.axes.get_xaxis().set_visible(False)
# f1.axes.get_yaxis().set_visible(False)
# plt.show()
# # plt.savefig('a1.png', bbox_inches='tight')

# cmap = plt.cm.jet
# norm = plt.Normalize(vmin=A.min(), vmax=A.max())

# image = cmap(norm(A))

# scipy.misc.imsave('aa.png', image)
# ***/

# Lấy index của các điểm thuộc cluster này
idx = np.where(pred == cluster_id)[0]

# Tạo lưới gồm 2 hàng, 5 cột và chiều cao là 10, chiều rộng là 4 với figsize
fig, ax = plt.subplots(2, 5, figsize=(10, 4))

# range dùng với index, enumerate dùng với list/array và có cả value và index
for i, ax in enumerate(ax.flat):          # axes.flat: duyệt qua từng ô trong lưới, dù lưới là 2D
    center_image = kmeans.cluster_centers_[i].reshape(28, 28)  # vector 784 -> ảnh 28x28
    ax.imshow(center_image, cmap='gray')  # vẽ ảnh vào ô đó
    ax.set_title(f'Cluster {i}')          # đặt tên là số cluster
    ax.axis('off')                        # ẩn trục x, y cho gọn 

plt.tight_layout()                        # Tự động canh chỉnh khoảng cách giữa các subplot
plt.savefig('image/clusters.png', bbox_inches='tight') 
plt.show()

# print(type(pred))
# print(pred.shape)
# print(type(X))