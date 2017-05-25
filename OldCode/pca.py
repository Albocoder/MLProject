#	FOR BETTER EXPLANATION CHECK THE LINK BELOW
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# x_train = np.loadtxt("./data/train/trainingData.txt")
# y_train = np.loadtxt("./data/train/trainingLabels.txt")
# x_test = np.loadtxt("./data/test/testData.txt")
# y_test = np.loadtxt("./data/test/testLabels.txt")

x = np.loadtxt("./data/subj1/data.txt")

pca = PCA(n_components=3)

pca.fit(x)

# maps the data to the given spcecified dimension
x_mapped = pca.transform(x)
x = None

np.savetxt("./data/subj1/data_pca.txt", x_mapped, fmt='%d')
x_mapped = None

plt.plot(pca.explained_variance_ratio_)
plt.xlabel('PCA index')
plt.ylabel('Explained Variance Ratio')
plt.show()
