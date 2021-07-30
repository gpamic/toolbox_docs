# 0. Import Libraries

import sys
sys.path.insert(0,'..')

from toolbox.datasets import classification_data_loader

from toolbox.preprocessing import hold_out
from toolbox.preprocessing.encoding import class_label_encode
from toolbox.preprocessing.normalization import normalize

from toolbox.evaluation.classification import confusion_matrix, compute_indices

from toolbox.learning.linear.ols import OrdinaryLeastSquare
from toolbox.learning.neural_networks.supervised.models.elm import ExtremeLearningMachine
from toolbox.learning.neural_networks.supervised.models.mlp import MultiLayerPerceptron
from toolbox.learning.svm.LSSVC import LSSVC
from toolbox.learning.svm.LSSVC_GPU import LSSVC_GPU

# 1. Load dataSet

X_raw, Y = classification_data_loader.load_dataset(dataset='iris')

# 2. Get dataset's info (number of samples, attributes, classes)

ds_info = classification_data_loader.extract_info(X_raw, Y)
N = ds_info['n']
p = ds_info['p']
n_classes = ds_info['n_classes']

# 3. Pre-process labels

Y = class_label_encode(Y, ds_info['n_classes'], label_type='binary')

# 4. Split Data Between Train and Test 

X_tr_raw, y_tr, X_ts_raw, y_ts = hold_out.random_subsampling(X = X_raw, Y = Y, 
                                                             train_size=0.8, seed=1)

# 5. Normalize data

X_tr = normalize(X_tr_raw, norm_type='z_score')
X_ts = normalize(X_ts_raw, norm_type='z_score', X_ref = X_tr_raw)

# 6. Model Build and Label Estimation

# OLS experimentation
ols = OrdinaryLeastSquare()
ols.fit(X_tr, y_tr, in_row=True)
y_ols = ols.predict(X_ts, in_row=True)

# ELM experimentation
elm = ExtremeLearningMachine()
elm.fit(X_tr, y_tr, in_row=True, h_layer_size=20)
y_elm = elm.predict(X_ts, in_row=True)

# MLP experimentation
NPL = [p, 4, n_classes]     # MLP layer structure
mlp = MultiLayerPerceptron()
mlp.fit(X=X_tr, Y=y_tr, Xv=X_tr, Yv=y_tr, NPL=NPL, epochs=20, 
        n_batches=1, alpha=1e-2, decay=0, momentum=0, l2=0, dropout_percent=0)
y_mlp = mlp.predict(X_ts)[-1]

# LSSVC experimentation
lssvc = LSSVC()
lssvc.fit(X_tr, y_tr)
y_lssvc = lssvc.predict(X_ts)

# LSSVC GPU experimentation
lssvc_gpu = LSSVC_GPU()
lssvc_gpu.fit(X_tr, y_tr)
y_lssvc_gpu = lssvc_gpu.predict(X_ts)

lssvc_gpu.dump(filepath='model', only_hyperparams=False)       # Test dumping

lssvc_gpu2   = LSSVC_GPU.load(filepath='model')                # Test loading
y_lssvc_gpu2 = lssvc_gpu2.predict(X_ts)

# 7. Evaluate Classifiers

print('Results - OLS')
cm_ols = confusion_matrix(y_ts, y_ols.T)
print(cm_ols)
print(compute_indices(cm_ols)[0])

print('Results - ELM')
cm_elm = confusion_matrix(y_ts, y_elm.T)
print(cm_elm)
print(compute_indices(cm_elm)[0])

print('Results - LSSVC')
cm_lssvc = confusion_matrix(y_ts, y_lssvc)
print(cm_lssvc)
print(compute_indices(cm_lssvc)[0])

print('Results - LSSVC (GPU)')
cm_lssvc_gpu = confusion_matrix(y_ts, y_lssvc_gpu)
print(cm_lssvc_gpu)
print(compute_indices(cm_lssvc_gpu)[0])

print('Results - LSSVC (GPU - Loaded)')
cm_lssvc_gpu2 = confusion_matrix(y_ts, y_lssvc_gpu2)
print(cm_lssvc_gpu2)
print(compute_indices(cm_lssvc_gpu2)[0])

print('Results - MLP')
cm_mlp = confusion_matrix(y_ts, y_mlp)
print(cm_mlp)
print(compute_indices(cm_mlp)[0])

#################################