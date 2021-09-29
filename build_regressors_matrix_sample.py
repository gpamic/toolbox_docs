# 0. Import Libraries

import sys
sys.path.insert(0,'..')

import numpy as np

from toolbox.datasets.buid_regressors_matrix import arx_regression_matrix_from_siso
from toolbox.datasets.buid_regressors_matrix import arx_regression_matrix_from_miso
from toolbox.datasets.buid_regressors_matrix import arx_regression_matrix_from_simo
from toolbox.datasets.buid_regressors_matrix import arx_regression_matrix_from_mimo

# #################################

# INPUT/OUTPUT SIGNALS

u1 = np.arange(0,50,1)
u2 = np.arange(50,100,1)
y1 = np.arange(100,150,1)
y2 = np.arange(150,200,1)

N = y1.shape[0]

print('Number of samples: ' + str(N))

# #################################

# SISO

print('SISO System:')

y_ts = y1

lag_y = 2

u_ts = u1

lag_u = 2

X,y = arx_regression_matrix_from_siso(u_ts,y_ts,lag_y,lag_u)

print(X.shape)
print(y.shape)

print(y[0:5])
print(X[0:5,:])

# #################################

# MISO

print('MISO System:')

y_ts = y1

lag_y = 2

u_ts = np.zeros((N,2))
u_ts[:,0] = u1
u_ts[:,1] = u2

lag_u = (2,2)

X,y = arx_regression_matrix_from_miso(u_ts,y_ts,lag_y,lag_u)

print(X.shape)
print(y.shape)

print(X[0:5,:])
print(y[0:5])

# #################################

# SIMO

print('SIMO System:')

y_ts = np.zeros((N,2))
y_ts[:,0] = y1
y_ts[:,1] = y2

lag_y = (2,2)

u_ts = u1

lag_u = 2

X,y = arx_regression_matrix_from_simo(u_ts,y_ts,lag_y,lag_u)

print(X[0:5,:])
print(y[0:5,:])

# #################################

# MIMO

print('MIMO System:')

y_ts = np.zeros((N,2))
y_ts[:,0] = y1
y_ts[:,1] = y2

lag_y = (2,2)

u_ts = np.zeros((N,2))
u_ts[:,0] = u1
u_ts[:,1] = u2

lag_u = (2,2)

X,y = arx_regression_matrix_from_mimo(u_ts,y_ts,lag_y,lag_u)

print(X[0:5,:])
print(y[0:5,:])

# #################################