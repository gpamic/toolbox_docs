from toolbox.evaluation.classifiers import confusion_matrix, compute_indices
from toolbox.learning.linear.ols import OrdinaryLeastSquare
from toolbox.learning.neural_networks.supervised.models.elm import ExtremeLearningMachine
from toolbox.learning.neural_networks.supervised.models.mlp import MultiLayerPerceptron
from toolbox.learning.svm.LSSVC import LSSVC
from toolbox.learning.svm.LSSVC_GPU import LSSVC_GPU
from toolbox.preprocessing.data_module import *
from toolbox.utils.dataset import dataset_pipeline

# Set Experiment

experiment = {
       'DS'  : 'iris',                                    # Dataset
       'CV'  : {'type': 'hold-out', 'train_size': 0.8},  # Hold-out cross validation with 5 repetitions Tr/Te = 80/20
       'PP'  : {'norm_type': 'z_score', 'label_encoding': 'binary'},
       'OUT' : {'file_name': 'teste_1', 'file_path': ''}
}

# Load Data Set

X_tr, y_tr, X_ts, y_ts = dataset_pipeline(s_info = experiment)

p = X_tr.shape[1]
n_classes = y_tr.shape[1]

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
mlp.fit(X=X_tr, Y=y_tr, Xv=X_tr, Yv=y_tr, NPL=NPL, epochs=20, n_batches=1, alpha=1e-2, decay=0, momentum=0, l2=0, dropout_percent=0)
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

# Get Results

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