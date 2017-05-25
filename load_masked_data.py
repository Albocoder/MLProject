####################################                 
####    Loading Haxby dataset   ####
####        subject 1        ####
####################################

#### Version    : 1.0
#### Date       : 20 May 2017

####### Import #########
from mvpa2.suite import *
import numpy as np
import os 
 
nLabel = 8
dir_path = os.path.dirname(os.path.realpath(__file__))
subjpath = os.path.join('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Dataset/subj1')
attrs = SampleAttributes(os.path.join(subjpath, 'labels.txt'),header=True)
ds = fmri_dataset(samples=os.path.join(subjpath, 'bold.nii.gz'),
                  targets=attrs.labels,
                  chunks=attrs.chunks,
                  mask=os.path.join(subjpath, 'mask4_vt.nii.gz'))
# preprocessing
poly_detrend(ds, polyord=1, chunks_attr='chunks')
zscore(ds, param_est=('targets', ['rest']), dtype='float32')

# delete rest samples
interesting = np.array([i in ['scissors', 'face', 'cat', 'shoe', 'house', 'scrambledpix', 'bottle', 'chair'] for i in ds.sa.targets])
ds = ds[interesting]

# extracts the label vector and assigns integers instead of class labels
ds.targets[ds.targets=='scissors'] = 0
ds.targets[ds.targets=='face'] = 1
ds.targets[ds.targets=='cat'] = 2
ds.targets[ds.targets=='shoe'] = 3
ds.targets[ds.targets=='house'] = 4
ds.targets[ds.targets=='scrambledpix'] = 5
ds.targets[ds.targets=='bottle'] = 6
ds.targets[ds.targets=='chair'] = 7
ds.targets = ds.targets.astype(int)



# partition the dataset
# divides the 12 chucks into 9 and 3 subsets
ds_train = ds[ds.chunks < 9]
ds_test = ds[ds.chunks >= 9]
del ds

# # create one hot encoding of train and test labels
# train_labels_onehot = np.zeros((ds_train.samples.shape[0], nLabel))
# for i in range(0, ds_train.samples.shape[0]):
# 	train_labels_onehot[i, ds_train.targets[i]] = 1

# test_labels_onehot = np.zeros((ds_test.samples.shape[0], nLabel))
# for i in range(0, ds_test.samples.shape[0]):
# 	test_labels_onehot[i, ds_test.targets[i]] = 1


# saves the files
f = file("./../data/subj1/data_masked_1Dy.bin","wb")
np.save(f,ds_train.samples)
np.save(f,ds_train.targets)
np.save(f,ds_test.samples)
np.save(f,ds_test.targets)
f.close()





