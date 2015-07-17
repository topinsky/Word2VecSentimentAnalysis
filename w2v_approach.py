# encoding=utf8
__author__ = 'topinsky'

import optparse
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier as SGDC

from utils import *


if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.add_option("--train", dest="train_file", metavar="FILE",
                      help="filename with reviews for training")
    parser.add_option("--test", dest="test_file", metavar="FILE",
                      help="filename with reviews for testing")
    parser.add_option("-b", "--model", dest="model_file", metavar="FILE",
                      help="bin filename with Word2Vec model")
    parser.add_option("--balanced", dest="balanced", action="store_true", default=False,
                      help="whether the script should balance classes' distribution.")
    (options, _) = parser.parse_args()

    if options.train_file is None:
        raise RuntimeError("Specify the input filename with train dataset, --train <filename>.")
    if options.targets_file is None:
        raise RuntimeError("Specify the input filename with test dataset, --test <filename>.")
    if options.model_file is None:
        raise RuntimeError("Specify the input bin file with model, -b <filename>.")

    print "Loading Word2Vec model..."
    model = Word2Vec.load_word2vec_format(options.model_file, binary=True)
    print "Loading datasets..."
    train_features, train_targets = load_data(options.train_file, model)
    test_features, test_targets = load_data(options.test_file, model)
    if options.balanced:
        print "Balancing..."
        train_index = balanced_index(train_targets)
        test_index = balanced_index(test_targets)
        train_features = train_features[train_index]
        test_features = test_features[test_index]
        train_targets = train_targets[train_index]
        test_targets = test_targets[test_index]

    logreg = SGDC(loss='log', n_iter=30, random_state=13, verbose=1, n_jobs=-2, class_weight='auto')

    # training
    print "Training..."
    logreg.fit(train_features, train_targets)

    # results

    print logreg.score(test_features, test_targets)

    expected = test_targets
    predicted = logreg.predict(test_features)

    print "Classification report for classifier %s:\n%s\n" % (logreg, classification_report(expected, predicted))
    print "Confusion matrix:\n%s" % confusion_matrix(expected, predicted)

    lr_res = predictions_and_stats(logreg, test_features, test_targets)
    roc(lr_res[1], lr_res[2], lr_res[3], "LogReg on W2V Model " + options.model_file)
    plt.show()
