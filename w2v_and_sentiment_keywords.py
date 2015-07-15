# encoding=utf8
__author__ = 'topinsky'

import optparse
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier as SGDC
from utils import *


if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.add_option("-m", "--matrix", dest="matrix_file", metavar="FILE",
                      help="filename with sparse feature matrix")
    parser.add_option("-t", "--targets", dest="targets_file", metavar="FILE",
                      help="filename with corresponding targets")
    parser.add_option("-k", "--keywords", dest="keywords_file", metavar="FILE",
                      help="filename with corresponding sentiment keywords")
    parser.add_option("-b", "--model", dest="model_file", metavar="FILE",
                      help="bin filename with Word2Vec model")
    parser.add_option("-s", "--split", dest="split", type=float, default=0.3,
                      help="Split ratio for constructing train/test split")
    parser.add_option("--balanced", dest="balanced", action="store_true", default=False,
                      help="whether the script should balance classes' distribution.")
    (options, _) = parser.parse_args()

    if options.matrix_file is None:
        raise RuntimeError("Specify the input filename with matrix, -m <filename>.")
    if options.targets_file is None:
        raise RuntimeError("Specify the input filename with targets, -t <filename>.")
    if options.keywords_file is None:
        raise RuntimeError("Specify the input filename with keywords, -k <filename>.")
    if options.model_file is None:
        raise RuntimeError("Specify the input bin file with model, -b <filename>.")

    print "Loading matrix and target data files..."
    matrix_train, matrix_test,  targets_train, targets_test = construct_baseline_datasets((options.matrix_file,
                                                                                           options.targets_file),
                                                                                          split_ratio=options.split,
                                                                                          balanced=options.balanced)
    print "Loading Word2Vec model..."
    model = Word2Vec.load_word2vec_format(options.model_file, binary=True)
    print "Loading sentiment keywords..."
    kw_df = pd.read_csv(options.keywords_file, sep='\t')
    is_neg = lambda x: x > 2460
    keyword_number = len(kw_df)
    keywords = kw_df.loc[:keyword_number/2-1, 'Keyword']
    print "Processing..."
    w2v_train = []
    for vec in matrix_train:
        elem = np.zeros(300)
        nelem = np.zeros(300)
        skipped = 0
        neg_count = 0
        pos_count = 0
        for ind in vec.nonzero()[1]:
            try:
                if is_neg(ind):
                    shifted = ind - keyword_number/2
                    nelem += model[keywords[shifted]]
                    neg_count += 1
                else:
                    elem += model[keywords[ind]]
                    pos_count += 1
            except KeyError:
                skipped += 1
        if pos_count:
            elem /= pos_count
        if neg_count:
            nelem /= neg_count
        w2v_train.append(np.hstack((elem, nelem)))
    w2v_test = []
    for vec in matrix_test:
        elem = np.zeros(300)
        nelem = np.zeros(300)
        skipped = 0
        neg_count = 0
        pos_count = 0
        for ind in vec.nonzero()[1]:
            try:
                if is_neg(ind):
                    shifted = ind - keyword_number/2
                    nelem += model[keywords[shifted]]
                    neg_count += 1
                else:
                    elem += model[keywords[ind]]
                    pos_count += 1
            except KeyError:
                skipped += 1
        if pos_count:
            elem /= pos_count
        if neg_count:
            nelem /= neg_count
        w2v_test.append(np.hstack((elem, nelem)))

    logreg = SGDC(loss='log', n_iter=30, random_state=13, verbose=1, n_jobs=-2, class_weight='auto')

    w2v_test = np.array(w2v_test)
    notnan_test = np.unique((np.isnan(w2v_test).astype(int)-1).nonzero()[0])
    w2v_train = np.array(w2v_train)
    notnan_train = np.unique((np.isnan(w2v_train).astype(int)-1).nonzero()[0])

    # training
    print "Training..."
    logreg.fit(w2v_train[notnan_train, :], targets_train[notnan_train])

    # results

    print logreg.score(w2v_test[notnan_test, :], targets_test[notnan_test])

    expected = targets_test
    predicted = logreg.predict(w2v_test[notnan_test, :])

    print "Classification report for classifier %s:\n%s\n" % (logreg, classification_report(expected, predicted))
    print "Confusion matrix:\n%s" % confusion_matrix(expected, predicted)

    lr_res = predictions_and_stats(logreg, w2v_test[notnan_test, :], targets_test[notnan_test])
    roc(lr_res[1], lr_res[2], lr_res[3], "LogReg on Processed Matrix (w2v plus sentiment keywords)")
    plt.show()
