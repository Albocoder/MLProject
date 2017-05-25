from mvpa2.suite import *
import numpy as np

subjpath = os.path.join('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Dataset/subj1')
attrs = SampleAttributes(os.path.join(subjpath, 'labels.txt'),header=True)
ds = fmri_dataset(samples=os.path.join(subjpath, 'bold.nii.gz'),
                  targets=attrs.labels,
                  chunks=attrs.chunks,
                  mask=os.path.join(subjpath, 'mask4_vt.nii.gz'))

poly_detrend(ds, polyord=1, chunks_attr='chunks')
zscore(ds, param_est=('targets', ['rest']), dtype='float32')

# subjpath = os.path.join('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Dataset/subj6')
# attrs = SampleAttributes(os.path.join(subjpath, 'labels.txt'),header=True)
# ds2 = fmri_dataset(samples=os.path.join(subjpath, 'bold.nii.gz'),
#                 targets=attrs.labels,
#                 chunks=attrs.chunks,
#                 mask=os.path.join('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Dataset/subj1/mask4_vt.nii.gz'))
#
# poly_detrend(ds2, polyord=1, chunks_attr='chunks')
# zscore(ds2, param_est=('targets', ['rest']), dtype='float32')


# # Code for the heatmap of the dataset for "smelling"
# ##############################################################333
# ds.samples = ds.samples.astype('float')
# pl.figure(figsize=(14, 6))
# pl.subplot(121)
# plot_samples_distance(ds, sortbyattr='chunks')
# pl.title('Distances: z-scored, detrended (sorted by chunks)')
# pl.subplot(122)
# plot_samples_distance(ds, sortbyattr='targets')
# pl.title('Distances: z-scored, detrended (sorted by targets)');
# pl.show()

## for generalizing trained model on subject1 with subject 6
# clf = SMLR()
# clf.train(ds)
# predictions = clf.predict(ds2.samples)
# print np.mean(predictions == ds.sa.targets)


# cross validation procedure
interesting = np.array([i in ['scissors', 'face', 'cat', 'shoe', 'house', 'scrambledpix', 'bottle', 'chair'] for i in ds.sa.targets])
ds = ds[interesting]
cv = CrossValidation(LinearNuSVMC(), OddEvenPartitioner(), enable_ca=['stats'])
error = cv(ds)
accuracy = np.mean(100*(np.array([1, 1]) - error))
print accuracy
print cv.ca.stats.as_string(description=True)


# # PyMVPA classifiers
# clfs = {
# 'Ridge Regression': RidgeReg(),
# 'Linear SVM': LinearNuSVMC(probability=1,
# enable_ca=['probabilities']),
# 'RBF SVM': RbfNuSVMC(probability=1,
# enable_ca=['probabilities']),
# 'SMLR': SMLR(lm=0.01),
# 'Logistic Regression': PLR(criterion=0.00001),
# '3-Nearest-Neighbour': kNN(k=3),
# '10-Nearest-Neighbour': kNN(k=10),
# 'GNB': GNB(common_variance=True),
# 'GNB(common_variance=False)': GNB(common_variance=False),
# 'LDA': LDA(),
# 'QDA': QDA(),
# }
