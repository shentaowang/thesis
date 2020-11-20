import numpy as np
import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from scipy.special import softmax
import joblib
from sklearn.metrics import accuracy_score


def train():
    affine_label_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/affine_label_ratio2/"
    tracker_predict_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-test-dev/tracker_iou_dists_2/"
    X = []
    y = []
    affine_thre = 40
    expand_dim = 100
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
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    shuffled_X, shuffled_y = shuffle(X, y)
    # train the data
    # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 16), random_state=1,  max_iter=10000)
    model = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(64, 16), random_state=1, max_iter=10000)
    model.fit(shuffled_X, shuffled_y)
    print(model.score(X, y))
    # calculate tp, fn, fp, fn
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
    joblib.dump(model, "mlp.pkl")


def test():
    affine_label_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/affine_label_ratio2/"
    tracker_predict_dir = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-val/tracker_iou_dists_2/"
    X = []
    y = []
    affine_thre = 40
    expand_dim = 100
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
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    # load the model
    model = joblib.load("mlp.pkl")
    predict_y = model.predict(X)
    print(accuracy_score(y, predict_y))


if __name__ == "__main__":
    train()
    test()
