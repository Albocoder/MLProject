#	FOR BETTER EXPLANATION CHECK THE LINK BELOW
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.cm as cm
import mpl_toolkits.mplot3d as axes3d

print('loading files...')
f = file("../data/subj1/data.bin","rb")
x_train = np.load(f)
y_train = np.load(f)
# load the test data
x_test = np.load(f)
y_test = np.load(f)
f.close()

x = np.concatenate((x_train, x_test), axis=0) 
labels = np.concatenate((y_train, y_test), axis=0) 

pca = PCA(n_components=3)

pca.fit(x)

# maps the data to the given spcecified dimension
data = pca.transform(x)
x = None





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
ax.scatter(data[index6, 0], data[index6, 1], data[index6, 2], color = 'b')
ax.scatter(data[index7, 0], data[index7, 1], data[index7, 2], color = 'b')
ax.scatter(data[index8, 0], data[index8, 1], data[index8, 2], color = 'k')



plt.show()
