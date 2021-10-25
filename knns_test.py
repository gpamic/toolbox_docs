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

from toolbox.learning.prototype_based.knns import knnSlidingWindowClassifier

##################### 1. Load dataSet

X_raw, Y = classification_data_loader.load_dataset(dataset='iris')

##################### 2. Get dataset's info (No of samples, attributes, classes)

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

knns = knnSlidingWindowClassifier(K=1, knn_type=1, Ws=6)

print('Hiperparametros:')
print(knns.K)
print(knns.knn_type)
print(knns.Ws)
input()

##################### 

acc = 0

for i in range(Ntr):

    # input("Verifica amostra " + str(i) + ":")
    yh = knns.partial_fit(X_tr2[:,i],y_tr2[:,i])
    
    if(np.argmax(yh) == np.argmax(y_tr2[:,i])):
        acc = acc + 1
    
    print('Amostra, predicao, acertos:')
    print(y_tr2[:,i])
    print(yh.T)
    print(acc)

    print('Parametros:')
    print(knns.Cx.shape)
    print(knns.Cy.shape)
    input('Wait!')
    
print(knns.Cx.shape)

acc_tr = acc / Ntr
print(acc_tr)

##################### 

y_knns = knns.predict(X_ts)

##################### 7. Evaluate Classifiers

print('\nResults - KNNs')
cm_knns = confusion_matrix(y_ts, y_knns.T)
print(cm_knns)
print(compute_indices(cm_knns)[0])

##################### 8. Visualize

plt.figure(figsize=(10,10))
plt.scatter(X[:,0], X[:,1], c='indigo')
plt.scatter(knns.Cx.T[:,0], knns.Cx.T[:,1], c='crimson')
plt.title(f'Dados e {knns.Cx.shape[1]} Protótipos', fontsize = 15)
plt.legend(['Dados', 'Prototipos'])

#################################















