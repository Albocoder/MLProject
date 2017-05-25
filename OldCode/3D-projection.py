import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.mplot3d as axes3d
import numpy as np

labels = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj1/labels.txt').astype(int)
data = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj1/data_pca3D.txt').reshape(1452, 3).astype(int)

index0 = np.array(np.flatnonzero(labels==0), dtype=int)
index1 = np.flatnonzero(labels==1)
index2 = np.flatnonzero(labels==2)
index3 = np.flatnonzero(labels==3)
index4 = np.flatnonzero(labels==4)
index5 = np.flatnonzero(labels==5)
index6 = np.flatnonzero(labels==6)
index7 = np.flatnonzero(labels==7)
index8 = np.flatnonzero(labels==8)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# ax.scatter(data[index0, 0], data[index0, 1], data[index0, 2], color = 'b')

ax.scatter(data[index1, 0], data[index1, 1], data[index1, 2], color = 'g')
ax.scatter(data[index2, 0], data[index3, 1], data[index2, 2], color = 'r')
ax.scatter(data[index3, 0], data[index3, 1], data[index3, 2], color = 'c')
ax.scatter(data[index4, 0], data[index4, 1], data[index4, 2], color = 'm')
ax.scatter(data[index5, 0], data[index5, 1], data[index5, 2], color = 'y')
# ax.scatter(data[index6, 0], data[index6, 1], data[index6, 2], color = 'b')
ax.scatter(data[index7, 0], data[index7, 1], data[index7, 2], color = 'b')
ax.scatter(data[index8, 0], data[index8, 1], data[index8, 2], color = 'k')





plt.show()
