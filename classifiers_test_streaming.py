####################  0. Import Libraries

import sys
sys.path.insert(0,'..')

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from toolbox.datasets import classification_data_loader

from toolbox.preprocessing import hold_out
from toolbox.preprocessing.encoding import class_label_encode
from toolbox.preprocessing.normalization import normalize
from toolbox.preprocessing.preprocess_sample import preprocess_input
from toolbox.preprocessing.preprocess_sample import preprocess_output

from toolbox.learning.prototype_based.prototypes_functions import prototypes_initialization
from toolbox.learning.prototype_based.prototypes_functions import prototypes_label

from toolbox.evaluation.classification import confusion_matrix, compute_indices

from toolbox.learning.prototype_based.wta import WinnerTakesAllClassifier
from toolbox.learning.prototype_based.knn import kNearestNeighbors

##################### 1. Load dataSet

X_raw, Y = classification_data_loader.load_dataset(dataset='iris')

##################### 2. Get dataset's info (number of samples, attributes, classes)

ds_info = classification_data_loader.extract_info(X_raw, Y)
N = ds_info['n_samples']
p = ds_info['n_inputs']
Nc = ds_info['n_outputs']

print('Samples: ' + str(N) + ' Attributes: ' + str(p) + ' Classes: ' + str(Nc))

##################### 3. Pre-process labels

Y = class_label_encode(Y, Nc, label_type='bipolar')

##################### 4. Split Data Between Train and Test 

Xtr_raw, y_tr, Xts_raw, y_ts = hold_out.random_subsampling(X = X_raw, 
                                                           Y = Y, 
                                                           train_size=0.8, 
                                                           random_state=1)
##################### 5. Normalize data

X_tr = normalize(Xtr_raw, norm_type='z_score')
X_ts = normalize(Xts_raw, norm_type='z_score', X_ref = Xtr_raw)
X = normalize(X_raw, norm_type='z_score',X_ref = Xtr_raw)

X_tr2 = preprocess_input(X_tr, in_row=True, bias=False)
y_tr2 = preprocess_output(y_tr, in_row=True)

N = X_tr2.shape[1]

X_ts2 = preprocess_input(X_ts, in_row=True, bias=False)
y_ts2 = preprocess_output(y_ts, in_row=True)

# # Verify data pattern

# print('Training dataset:')
# print(X_tr.shape)
# print(X_tr[0:5,:])
# print('')

# print('Test dataset:')
# print(y_tr.shape)
# print(y_tr[0:5,:])
# print('')

##################### 6. Model Build and Label Estimation

# WTA experimentation

wta = WinnerTakesAllClassifier(Nprot=10, K=1)

Nep = 200
tmax = N*Nep

wta.Cx = prototypes_initialization(X_tr2[:,1],wta.Nk)

for epoch in range(Nep):
    
    # Shuffle data
    I = np.random.permutation(N)
    Xrand = X_tr2[:,I]

    # Update Prototypes
    for i in range(N):
        wta.partial_fit(Xrand[:,i],tmax)

wta.Cy = prototypes_label(X_tr2, y_tr2, wta.Cx, wta.lbl, wta.dist)

y_wta = wta.predict(X_ts)

# WTA visualization
plt.figure(figsize=(10,10))
plt.scatter(X[:,0], X[:,1], c='indigo')
plt.scatter(wta.Cx.T[:,0], wta.Cx.T[:,1], c='crimson')
plt.title(f'Dados e {wta.Nk} Protótipos', fontsize = 15)
plt.legend(['Dados', 'Prototipos'])

# KNN experimentation
knn = kNearestNeighbors(K=2)
knn.fit(X_tr,y_tr)
y_knn = knn.predict(X_ts)

##################### 7. Evaluate Classifiers

print('Results - WTA')
cm_wta = confusion_matrix(y_ts, y_wta.T)
print(cm_wta)
print(compute_indices(cm_wta)[0])

print('Results - KNN')
cm_knn = confusion_matrix(y_ts, y_knn.T)
print(cm_knn)
print(compute_indices(cm_knn)[0])

#################################