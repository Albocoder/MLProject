from mvpa2.suite import *
import numpy as np

def shuffle_rows(arr,rows):
    np.random.shuffle(arr[rows[0]:rows[1]+1])

# loads the files
data = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/data_pca.txt').reshape(1452, 500).astype(int)
labels = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/labels.txt').reshape(1452, 1).astype(int)

# combines the data and labels
dataLabels = np.concatenate((data, labels), axis=1)
lastColIndex = len(dataLabels[0,:]) -1

# remove rest samples
zeroRows = np.flatnonzero(dataLabels[:, lastColIndex]==0)
dataLabels = np.delete(dataLabels, zeroRows, axis=0)
lastRowIndex = len(dataLabels[:,0]) - 1

# shuffles the samples(rows)
shuffle_rows(dataLabels, [0, lastRowIndex])


# partitioning the data
# 20% test data
training = dataLabels[:0.8*lastRowIndex,:]
test = dataLabels[0.8*lastRowIndex:,:]

# 20% of 80% = 16% validation and 64% training
lastRowIndexTrain = len(training[:,0]) - 1
validation = training[:0.2*lastRowIndexTrain,:]
training = training[0.2*lastRowIndexTrain:,:]


# dividing into labels and data
trainingData = training[:, 0:lastColIndex]
trainingLabels = training[:, lastColIndex]

validationData = validation[:, 0:lastColIndex]
validationLabels = validation[:, lastColIndex]

testData = test[:, 0:lastColIndex]
testLabels = test[:, lastColIndex]


# saves the files
np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/train/trainingData.txt', trainingData, fmt='%d')
np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/train/trainingLabels.txt', trainingLabels, fmt='%d')

np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/validation/validationData.txt', validationData, fmt='%d')
np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/validation/validationLabels.txt', validationLabels, fmt='%d')

np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/test/testData.txt', testData, fmt='%d')
np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/test/testLabels.txt', testLabels, fmt='%d')
