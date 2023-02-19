# import numpy as np
# import torch

# from kmean_manhattan import KMeansManhattan

# # Generate a 2-dimensional dataset with 100 points and 4 clusters
# np.random.seed(123)
# X0 = np.random.uniform(size=(200, 2))
# X0 = torch.tensor(X0[:-2].reshape(-1,3))

# dic = {}
# dic[0] = X0
# K = 3


# for iter in range(4):
#     for i in range(K):
#         kmeans = KMeansManhattan(np.array(dic[-1]), K=K, print=True)
#         labels, centers, best_sse = kmeans.process()

#         for k, value in enumerate(centers):
#             dic[k] = X0[labels == k]


# for i in range(len(dic)):
#     kmeans = KMeansManhattan(np.array(dic[i]), K=K, print=True)
#     labels, centers, best_sse = kmeans.process()

#     X2 = {}
#     for k, value in enumerate(centers):
#         X2[k] = dic[i][labels == k]
#     dic[i] = X2


# print(labels)
# print(centers)
# print(best_sse)

# # Plot the clustering results
# import matplotlib.pyplot as plt
# # plt.scatter(X[:, 0], X[:, 1], c=labels)
# # plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
# # plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2], c=labels)
# ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x')
# plt.show()

import numpy as np
import torch

from kmean_manhattan import KMeansManhattan

# Generate a 2-dimensional dataset with 100 points and 4 clusters
np.random.seed(123)
X0 = np.random.uniform(size=(200, 2))
X0 = torch.tensor(X0[:-2].reshape(-1,3))

K = 3

kmeans = KMeansManhattan(np.array(X0), K=K, print=True)
labels, centers, best_sse = kmeans.process()

list = []
for k, value in enumerate(centers):
    cluster = X0
    list.append((torch.tensor(value), cluster[labels == k]))

for i in range(len(centers)):
    cluster = list[i][1]
    kmeans = KMeansManhattan(np.array(cluster), K=K, print=True)
    labels, centers, best_sse = kmeans.process()

    temp_list = []
    for k, value in enumerate(centers):
        temp_list.append((torch.tensor(value), cluster[labels == k]))
    list[i] = (list[i][0], temp_list)

# clos_clus1 = closest(for i in range(len(centers)) return only closest(list[i][0]))
# clos_clus2 = closest(for j in range(len(centers)) return only closest(list[idx(clos_clus1)][1][0]))
# clos_row = return closest row in list[idx(clos_clus1)][1][idx(clos_clus2)][1]


print(labels)
print(centers)
print(best_sse)

# Plot the clustering results
import matplotlib.pyplot as plt
# plt.scatter(X[:, 0], X[:, 1], c=labels)
# plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2], c=labels)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x')
plt.show()