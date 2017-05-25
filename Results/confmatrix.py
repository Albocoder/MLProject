from __future__ import  division
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def plotConfMatrix(theMatrix,theClasses,title='The Confusion matrix',colorRange=plt.cm.GnBu):
    # creating the image to show using the data and chosen color bar
    plt.imshow(theMatrix,cmap=colorRange,interpolation='nearest')
    plt.colorbar()
    # creating tick marks as many as the classes in this case 3
    tick_marks = np.arange(len(theClasses))
    # adding a title
    plt.title(title)
    # creating x tickmarks
    plt.xticks(tick_marks, theClasses)
    plt.xlabel('Predicted label')
    # creating y tickmarks
    plt.yticks(tick_marks, theClasses)
    plt.ylabel('True label')
    thresh = theMatrix.max() / 2.
    for i, j in itertools.product(range(theMatrix.shape[0]), range(theMatrix.shape[1])):
        plt.text(j, i, theMatrix[i, j],
                 horizontalalignment="center",
                 color="white" if theMatrix[i, j] > thresh else "black")
    # show what you plotted to the world
    plt.show()

confmatrix = np.loadtxt('outputConfMatrix0.txt')
plotConfMatrix(confmatrix,['scissors','face','cat','shoe','house','scr.pix','bottle','chair'],"SVM with Linear kernel")
