#####################  

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:26:31 2021

@author: david
"""

####################  0. Import Libraries

import sys
sys.path.insert(0,'..')

import numpy as np

#######

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

#######

from toolbox.datasets import classification_data_loader

from toolbox.preprocessing import hold_out
from toolbox.preprocessing.encoding import class_label_encode
from toolbox.preprocessing.normalization import normalize
from toolbox.preprocessing.preprocess_sample import preprocess_input
from toolbox.preprocessing.preprocess_sample import preprocess_output

#######

from toolbox.evaluation.classification import confusion_matrix, compute_indices

#######

from toolbox.learning.prototype_based.spok import SpokNnClassifier

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
                                                           random_state=7)
##################### 5. Normalize data

X_tr = normalize(Xtr_raw, norm_type='z_score')
X_ts = normalize(Xts_raw, norm_type='z_score', X_ref = Xtr_raw)
X = normalize(X_raw, norm_type='z_score',X_ref = Xtr_raw)

X_tr2 = preprocess_input(X_tr, in_row=True, bias=False)
y_tr2 = preprocess_output(y_tr, in_row=True)

Ntr = X_tr2.shape[1]

X_ts2 = preprocess_input(X_ts, in_row=True, bias=False)
y_ts2 = preprocess_output(y_ts, in_row=True)

# Verify data pattern

print('Training dataset:')
print(X_tr.shape)
print(X_tr[0:5,:])
print('')

print('Test dataset:')
print(y_tr.shape)
print(y_tr[0:5,:])
print('')

##################### 6. Model Build and Label Estimation

### PROTOTYPE BASED

# Nk = 9
# Nneig = 1

# # KNN experimentation
# knn = kNearestNeighbors(K=2)
# knn.fit(X_tr,y_tr)
# y_knn = knn.predict(X_ts)

##################### 

spok = SpokNnClassifier(v1 = 0.99, gamma = 4)

print('Hiperparametros:')
print(spok.Dm)
print(spok.Ss)
print(spok.v1)
print(spok.v2)
print(spok.Us)
print(spok.eta)
print(spok.Ps)
print(spok.min_score)
print(spok.max_prot)
print(spok.min_prot)
print(spok.Von)
print(spok.K)
print(spok.knn_type)
print(spok.Kname)
print(spok.sig2n)
print(spok.kernel_params)

##################### 

acc = 0

for i in range(Ntr):

    # input("Verifica amostra " + str(i) + ":")
    yh = spok.partial_fit(X_tr2[:,i],y_tr2[:,i])
    
    if(np.argmax(yh) == np.argmax(y_tr2[:,i])):
        acc = acc + 1
    
    # print('Amostra, predicao, acertos:')
    # print(y_tr2[:,i])
    # print(yh.T)
    # print(acc)

    # print('Parametros: ')
    # print(spok.Cx.shape)
    # print(spok.Cy.shape)
    # print(spok.Km.shape)
    # print(spok.Kinv.shape)
    # print(spok.Kmc)
    # print(spok.Kinvc)
    # print(spok.score)
    # print(spok.class_history)
    # print(spok.times_selected)
    # print(np.dot(spok.Km,spok.Kinv))

# print(spok.Cx.shape)

acc_tr = acc / Ntr
print(acc_tr)

##################### 

y_spok = spok.predict(X_ts)

##################### 7. Evaluate Classifiers

print('\nResults - SPOK')
cm_spok = confusion_matrix(y_ts, y_spok.T)
print(cm_spok)
print(compute_indices(cm_spok)[0])

##################### 8. Visualize

plt.figure(figsize=(10,10))
plt.scatter(X[:,0], X[:,1], c='indigo')
plt.scatter(spok.Cx.T[:,0], spok.Cx.T[:,1], c='crimson')
plt.title(f'Dados e {spok.Cx.shape[1]} Protótipos', fontsize = 15)
plt.legend(['Dados', 'Prototipos'])

#################################















