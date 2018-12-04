from scipy.io import loadmat
train_idxs = loadmat('cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()

import json
with open('feature_data.json', 'r') as f:
    features = json.load(f)
