from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



print('loading files...')
f = file("../data/subj1/data.bin","rb")
x_train = np.load(f)
y_train = np.load(f).astype(int)
# load the test data
x_test = np.load(f)
y_test = np.load(f).astype(int)
f.close()


C = 3.0  # SVM regularization parameter
classifiers = []
# classifiers.append(svm.SVC(kernel='linear', C=C).fit(x_train, y_train))
# classifiers.append(svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x_train, y_train))
# classifiers.append(svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train))
classifiers.append(svm.LinearSVC(C=C).fit(x_train, y_train))
# for i in range(1, 10, 2):
#     classifiers.append(KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train))
# classifiers.append(tree.DecisionTreeClassifier().fit(x_train, y_train))
# classifiers.append(GaussianNB().fit(x_train, y_train))
for i, clf in enumerate(classifiers):
    P = clf.predict(x_test)
    CM = confusion_matrix(y_test,P)
    np.savetxt('./Results/outputConfMatrix'+str(i)+'.txt',CM,fmt='%d')
    np.savetxt('./Results/outputAccRate'+str(i)+'.txt',[np.sum([y_test==P])/len(y_test)])
