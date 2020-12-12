import os
import pickle
import numpy as np
from scipy.special import softmax


def load_data(affine_label_dir, tracker_predict_dir, affine_thre, expand_dim):
    X = []
    y = []
    # prepared the data
    for file in os.listdir(affine_label_dir):
        fin = open(os.path.join(affine_label_dir, file), 'rb')
        affine_label = pickle.load(fin)
        fin = open(os.path.join(tracker_predict_dir, file), 'rb')
        tracker_predict = pickle.load(fin)
        gt_labels = []
        pooling_data = []
        for key in affine_label:
            if affine_label[key]['dist'] > affine_thre:
                gt_labels.append(1)
            else:
                gt_labels.append(0)
        for key in tracker_predict:
            dists = tracker_predict[key]
            dists = np.concatenate([dists, np.ones((dists.shape[0], expand_dim - dists.shape[1]))], axis=1)
            dists = np.concatenate([dists, np.ones((expand_dim - dists.shape[0], expand_dim))], axis=0)
            dists = 10 - dists * 10
            pooling_dim0 = np.max(softmax(dists, axis=0), axis=0)
            pooling_dim1 = np.max(softmax(dists, axis=1), axis=1)
            pooling_data.append(np.concatenate([pooling_dim0, pooling_dim1]))
        y.extend(gt_labels)
        X.extend(pooling_data[1:])
    return X, y
