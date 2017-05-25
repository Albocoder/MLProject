from mvpa2.suite import *
import numpy as np

# prepares the fmri_dataset prameters
subjpath = os.path.join('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Dataset/subj1')
attrs = SampleAttributes(os.path.join(subjpath, 'labels.txt'),header=True)

# loads the fmri file into and ds object with all the attributes
ds = fmri_dataset(samples=os.path.join(subjpath,'bold.nii.gz'),
targets=attrs.labels, chunks=attrs.chunks,
mask=os.path.join(subjpath, 'mask4_vt.nii.gz'))




# next only works with floating point data
ds.samples = ds.samples.astype('float')

# look at sample similarity
# Note, the decreasing similarity with increasing temporal distance
# of the samples

# # preprocessing
poly_detrend(ds, polyord=1, chunks_attr='chunks')
zscore(ds, param_est=('targets', ['rest']), dtype='float32')

pl.title('Sample distances (before preprocessing)')
plot_samples_distance(ds, sortbyattr='chunks')

pl.show()
