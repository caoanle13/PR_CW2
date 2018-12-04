# 13,164 images in data set
# 1,360 pedestrians

import numpy as np

from scipy.io import loadmat
camId = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
filelist = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()
gallery_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
labels = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
train_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
query_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()

train_idx -= 1
query_idx -= 1



import json
with open('PR_data/feature_data.json', 'r') as f:
    features = np.array(json.load(f))

X_train = features[train_idx]
y_train = labels[train_idx]
X_test = features[query_idx]
y_test = labels[query_idx]










