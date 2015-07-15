# encoding=utf8
__author__ = 'topinsky'
import matplotlib
matplotlib.use('TkAgg')
import csv
import numpy as np
import pylab as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_score, roc_curve, auc
from sklearn.utils import resample
from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE

sns.set()


def predictions_and_stats(model, test_features, test_targets):
    pred_probas = model.predict_proba(test_features)[:, 1]
    predicts = model.predict(test_features)
    print "Avg.Accuracy: %.3f" % model.score(X=test_features,
                                             y=test_targets)
    print "Avg.Precision: %.3f" % precision_score(y_true=test_targets,
                                                  y_pred=predicts)
    fpr, tpr, thrs = roc_curve(y_true=test_targets, y_score=pred_probas)
    roc_auc = auc(fpr, tpr)

    tp, tn, fp, fn = [], [], [], []

    thrs = thrs[::-1]

    p = sum(test_targets)
    n = len(test_targets) - p
    print "Positives: %d (%.2f%%), Negatives: %d (%.2f%%)" % (p, 100.*p/(p+n), n, 100.*n/(p+n))
    for i in range(len(thrs)):
        tp.append(p * tpr[i])
        fn.append(p * (1. - tpr[i]))
        fp.append(n * fpr[i])
        tn.append(n * (1. - fpr[i]))

    tp = np.array(tp)[::-1]
    tn = np.array(tn)[::-1]
    fp = np.array(fp)[::-1]
    fn = np.array(fn)[::-1]
    avails = (tp + fp)/(n+p)
    avails2 = tp / (tp + fn)
    precision = tp / (tp + fp + 0.00001)
    return thrs, fpr, tpr, roc_auc, precision, avails, avails2


def predictions_and_stats2(model, test_features, test_targets):
    pred_probas = model.decision_function(test_features)
    predicts = model.predict(test_features)
    print "Avg.Accuracy: %.3f" % model.score(X=test_features,
                                             y=test_targets)
    print "Avg.Precision: %.3f" % precision_score(y_true=test_targets,
                                                  y_pred=predicts)
    fpr, tpr , thrs = roc_curve(y_true=test_targets, y_score=pred_probas)
    roc_auc = auc(fpr, tpr)

    tp, tn, fp, fn = [], [], [], []

    thrs = thrs[::-1]

    p = sum(test_targets)
    n = len(test_targets) - p
    print "Positives: %d (%.2f%%), Negatives: %d (%.2f%%)" % (p, 100.*p/(p+n), n, 100.*n/(p+n))
    for i in range(len(thrs)):
        tp.append(p * tpr[i])
        fn.append(p * (1. - tpr[i]))
        fp.append(n * fpr[i])
        tn.append(n * (1. - fpr[i]))

    tp = np.array(tp)[::-1]
    tn = np.array(tn)[::-1]
    fp = np.array(fp)[::-1]
    fn = np.array(fn)[::-1]
    avails = (tp + fp)/(n+p)
    avails2 = tp / (tp + fn)
    precision = tp / (tp + fp + 0.00001)
    return thrs, fpr, tpr, roc_auc, precision, avails, avails2


def roc(fpr,tpr,roc_auc,title):
    plt.plot(fpr, tpr, label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(title)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc='lower right')


def kpi(thrs, precision, avails,  title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thrs*100,precision,'-', color='red', label = 'Precision', alpha=0.3)
    ax.plot(thrs*100,avails,'-', color='blue', label = 'Avails', alpha =0.3)
    old_point = None
    for i,point in enumerate(zip(thrs*100,precision)):
        if i > 0 and (np.linalg.norm(np.array(point) - old_point) < 9):
            continue
        ax.annotate(str(int(precision[i]*100))+'%', xy=np.array(point) + np.array([0.01,-0.05]), fontsize = 14, color='green')
        ax.plot([point[0]],[point[1]],'ok')
        old_point = np.array(point)
    for i,point in enumerate(zip(thrs*100,avails)):
        if i > 0 and (np.linalg.norm(np.array(point) - old_point) < 9):
            continue
        ax.annotate(str(int(avails[i]*100))+'%', xy=np.array(point) + np.array([0.01,0.02]), fontsize = 14, color='green')
        ax.plot([point[0]],[point[1]],'ok')
        old_point = np.array(point)

    plt.title(title)
    plt.xlim([0.0, 100.])
    plt.xticks(np.arange(0,101,10))
    plt.ylim([0.0, 1.05])
    ax.set(yticks = np.arange(0,1.01,.10), yticklabels = np.arange(0,101,10))
    plt.xlabel('cut-off')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)


def construct_baseline_datasets(filenames, split_ratio=0.3, balanced=False):
    # initial matrix file: '/mnt/hadoop/Users/Eswar/Sentiment/yelpAndAmazon/matrix_sparse.txt'
    # initial target file: '/mnt/hadoop/Users/Eswar/Sentiment/yelpAndAmazon/vector_b.txt'
    csvreader = csv.reader(open(filenames[0]))
    rdim, cdim, _ = map(int, next(csvreader))
    rows, columns, data = [], [], []
    for line in csvreader:
        row, column, count = map(int, line)
        rows.append(row-1)
        columns.append(column-1)
        data.append(count)
    matrix = csr_matrix((data, (rows, columns)), shape=(rdim, cdim))
    csvreader = csv.reader(open(filenames[1]))
    targets = []
    for line in csvreader:
        t = map(int, line)
        targets.append(t[0])
    targets = np.array(targets)
    known_sentiment_indices = targets.nonzero()[0]
    i2 = (targets[known_sentiment_indices] + 1)/2

    if balanced:
        """ It's assumed that positive class overpopulates the data set and
          the zero class forms minority."""
        class_index = {0: [], 1: []}
        for i, c in enumerate(i2):
            class_index[c].append(i)
        balanced_class_index = resample(class_index[1],
                                        n_samples=len(class_index[0]),
                                        replace=False,
                                        random_state=5)
        balanced_class_index.sort()
        balanced_index = np.concatenate((class_index[0], balanced_class_index))
        balanced_index.sort()
        matrix = matrix[known_sentiment_indices]
        matrix_train, matrix_test, targets_train, targets_test = train_test_split(matrix[balanced_index],
                                                                                  i2[balanced_index], test_size=split_ratio)
    else:
        matrix_train, matrix_test, targets_train, targets_test = train_test_split(matrix[known_sentiment_indices, :],
                                                                                  i2, test_size=split_ratio)
    return matrix_train, matrix_test,  targets_train, targets_test


def tsne_plot(features, targets):
    ts_all = TSNE(2)
    subsmpl = resample(range(len(features)),
                       n_samples=2000,
                       replace=False,
                       random_state=5)
    reduced_vecs = ts_all.fit_transform(features[subsmpl, :])
    targets = np.array(targets)
    targets_smpl = targets[subsmpl]

    # color points by class label to see if Word2Vec can separate them
    for i in range(len(reduced_vecs)):
        if targets_smpl[i]:
            # positive
            color = 'g'
        else:
            # negative
            color = 'r'
        plt.plot(reduced_vecs[i,0], reduced_vecs[i,1], marker='o', color=color, markersize=8, alpha =0.3)


if __name__ == '__main__':
    pass
