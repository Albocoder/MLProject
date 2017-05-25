from mvpa2.suite import *
import numpy as np

def shuffle_rows(arr,rows):
    np.random.shuffle(arr[rows[0]:rows[1]+1])

# loads the files
data1 = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj1/unmasked/data.txt').reshape(1452, 163840).astype(int)
labels1 = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj1/unmasked/labels.txt').reshape(1452, 1).astype(int)
dataLabels1 = np.concatenate((data1, labels1), axis=1)
data1 = None
labels1 = None

data2 = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj2/unmasked/data.txt').reshape(1452, 163840).astype(int)
labels2 = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj2/unmasked/labels.txt').reshape(1452, 1).astype(int)
dataLabels2 = np.concatenate((data2, labels2), axis=1)
data2 = None
labels2 = None

dataLabels2 = np.concatenate((dataLabels1, dataLabels2), axis=0)
dataLabels1 = None

data3 = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj3/unmasked/data.txt').reshape(1452, 163840).astype(int)
labels3 = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj3/unmasked/labels.txt').reshape(1452, 1).astype(int)
dataLabels3 = np.concatenate((data3, labels3), axis=1)
data3 = None
labels3 = None

dataLabels3 = np.concatenate((dataLabels2, dataLabels3), axis=0)
dataLabels2 = None

data4 = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj4/unmasked/data.txt').reshape(1452, 163840).astype(int)
labels4 = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj4/unmasked/labels.txt').reshape(1452, 1).astype(int)
dataLabels4 = np.concatenate((data4, labels4), axis=1)
data4 = None
labels4 = None

dataLabels4 = np.concatenate((dataLabels3, dataLabels4), axis=0)
dataLabels3 = None

data5 = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj5/unmasked/data.txt').reshape(1452, 163840).astype(int)
labels5 = np.loadtxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/subj5/unmasked/labels.txt').reshape(1452, 1).astype(int)
dataLabels5 = np.concatenate((data5, labels5), axis=1)
data5 = None
labels5 = None

dataLabels5 = np.concatenate((dataLabels4, dataLabels5), axis=0)
dataLabels4 = None

# removes the rest samples
lastColIndex = len(dataLabels5[0,:]) - 1
zeroRows = np.flatnonzero(dataLabels5[:, lastColIndex]==0)
dataLabels5 = np.delete(dataLabels5, zeroRows, axis=0)
lastRowIndex = len(dataLabels5[:,0]) - 1

# shuffles the samples(rows)
shuffle_rows(dataLabels5, [0, lastRowIndex])

# partitioning the data
# 20% test data
training = dataLabels5[:0.8*lastRowIndex,:]
test = dataLabels5[0.8*lastRowIndex:,:]
# 20% of 80% = 16% validation and 64% training
lastRowIndexTrain = len(training[:,0]) - 1
validation = training[:0.2*lastRowIndexTrain,:]
training = training[0.2*lastRowIndexTrain:,:]


# dividing into labels and data
trainingData = training[:, 0:lastColIndex]
trainingLabels = training[:, lastColIndex]
training = None

validationData = validation[:, 0:lastColIndex]
validationLabels = validation[:, lastColIndex]
validation = None

testData = test[:, 0:lastColIndex]
testLabels = test[:, lastColIndex]
test = None


# saves the files
np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/train/trainingData.txt', trainingData, fmt='%d')
np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/train/trainingLabels.txt', trainingLabels, fmt='%d')

np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/validation/validationData.txt', validationData, fmt='%d')
np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/validation/validationLabels.txt', validationLabels, fmt='%d')

np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/test/testData.txt', testData, fmt='%d')
np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/test/testLabels.txt', testLabels, fmt='%d')

trainingData = None
trainingLabels = None

validationData = None
validationLabels = None

testData = None
testLabels = None


# dividing into labels and data
data = dataLabels5[:, 0:lastColIndex]
labels = dataLabels5[:, lastColIndex]
dataLabels5 = None

# saves the files
np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/data.txt', data, fmt='%d')
np.savetxt('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Project/Codes/MLProject/data/lebels.txt', labels, fmt='%d')
