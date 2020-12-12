import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy.special import softmax
import joblib
from utils.data import load_data


def train():
    affine_thre = 10
    expand_dim = 200
    affine_label_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/" \
                       "affine_label_surf_ratio2_p1/"
    tracker_predict_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/" \
                          "tracker_extend_iou_dists_2_det0.3_1204_p1/"
    X, y = load_data(affine_label_dir, tracker_predict_dir, affine_thre, expand_dim)
    affine_label_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/" \
                       "affine_label_surf_ratio2_p2/"
    tracker_predict_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/" \
                          "tracker_extend_iou_dists_2_det0.3_1204_p2/"
    add_X, add_y = load_data(affine_label_dir, tracker_predict_dir, affine_thre, expand_dim)
    X.extend(add_X)
    y.extend(add_y)
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    shuffled_X, shuffled_y = shuffle(X, y)    # train the data
    model = LogisticRegression(solver='liblinear')
    model.fit(shuffled_X, shuffled_y)

    tp, fn, fp, tn = 0, 0, 0, 0
    predict_y = model.predict(X)
    for i in range(len(y)):
        if y[i] == 1 and predict_y[i] == 1:
            tp += 1
        elif y[i] == 1 and predict_y[i] == 0:
            fn += 1
        elif y[i] == 0 and predict_y[i] == 1:
            fp += 1
        else:
            tn += 1
    accuracy_train = accuracy_score(y, predict_y)
    print(accuracy_train)
    print("tp: {}, fn: {}, fp: {}, tn: {}".format(tp, fn, fp, tn))
    joblib.dump(model, "model/regression.pkl")


def test():
    affine_thre = 10
    expand_dim = 200
    affine_label_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/" \
                       "affine_label_surf_ratio2_p1/"
    tracker_predict_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/" \
                          "tracker_extend_iou_dists_2_det0.3_1204_p1/"
    X, y = load_data(affine_label_dir, tracker_predict_dir, affine_thre, expand_dim)
    affine_label_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/" \
                       "affine_label_surf_ratio2_p2/"
    tracker_predict_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/" \
                          "tracker_extend_iou_dists_2_det0.3_1204_p2/"
    add_X, add_y = load_data(affine_label_dir, tracker_predict_dir, affine_thre, expand_dim)
    X.extend(add_X)
    y.extend(add_y)
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    # load the model
    model = joblib.load("model/regression.pkl")
    print(model)
    predict_y = model.predict(X)
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == 1 and predict_y[i] == 1:
            tp += 1
        elif y[i] == 1 and predict_y[i] == 0:
            fn += 1
        elif y[i] == 0 and predict_y[i] == 1:
            fp += 1
        else:
            tn += 1
    print("tp: {}, fn: {}, fp: {}, tn: {}".format(tp, fn, fp, tn))
    print(accuracy_score(y, predict_y))


if __name__ == "__main__":
    train()
    test()
