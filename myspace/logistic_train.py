import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy.special import softmax


def train():
    affine_label_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/affine_label_ratio2/"
    tracker_predict_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/tracker_iou_dists_2/"
    X = []
    y = []
    affine_thre = 30
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
            dists = np.concatenate([dists, np.ones((dists.shape[0], 200 - dists.shape[1]))], axis=1)
            dists = np.concatenate([dists, np.ones((200 - dists.shape[0], 200))], axis=0)
            dists = 10 - dists * 10
            pooling_dim0 = np.max(softmax(dists, axis=0), axis=0)
            pooling_dim1 = np.max(softmax(dists, axis=1), axis=1)
            pooling_data.append(np.concatenate([pooling_dim0, pooling_dim1]))
        y.extend(gt_labels)
        X.extend(pooling_data[1:])
    X = np.array(X)
    y = np.array(y)
    shuffled_X, shuffled_y = shuffle(X, y)
    # train the data
    model = LogisticRegression(solver='liblinear')
    model.fit(shuffled_X, shuffled_y)

    predict_train = model.predict(X)
    accuracy_train = accuracy_score(y, predict_train)
    print(accuracy_train)


if __name__ == "__main__":
    train()