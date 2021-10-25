# 0. Import Libraries

import sys
sys.path.insert(0,'..')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from toolbox.datasets import classification_data_loader

# from toolbox.preprocessing import hold_out
# from toolbox.preprocessing.encoding import class_label_encode
from toolbox.preprocessing.normalization import normalize

from toolbox.learning.prototype_based.wta import WinnerTakesAllClustering

from toolbox.learning.prototype_based.wta import WinnerTakesAllClassifier

# from toolbox.evaluation.clustering import compute_indices

# 1. Load dataSet

X_raw, Y = classification_data_loader.load_dataset(dataset='iris')

# 2. Get dataset's info (number of samples, attributes, classes)

ds_info = classification_data_loader.extract_info(X_raw, Y)
N = ds_info['n_samples']
p = ds_info['n_inputs']
Nc = ds_info['n_outputs']

print('Samples: ' + str(N) + ' Attributes: ' + str(p) + ' Classes: ' + str(Nc))

# 5. Normalize data

X_norm = normalize(X_raw, norm_type = 'z_score', X_ref = X_raw)

# 6. Build Model

Nk = 5 # Define number of prototypes

wtaCluster = WinnerTakesAllClustering(Nprot = Nk)
wtaCluster.fit(X_norm)
indices = wtaCluster.predict(X_norm)

wtaClass = WinnerTakesAllClassifier(Nprot = Nk)
wtaClass.fit(X_norm,Y)

# 7. Visualize

print(indices.shape)
print(indices)

plt.figure(figsize=(10,10))
plt.scatter(X_norm[:,0], X_norm[:,1], c='indigo')
plt.scatter(wtaCluster.Cx.T[:,0], wtaCluster.Cx.T[:,1], c='crimson')
plt.title(f'Dados e {Nk} Protótipos', fontsize = 15)
plt.legend(['Dados', 'Prototipos'])

# 8. Evaluate

#################################