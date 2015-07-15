# encoding=utf8
__author__ = 'topinsky'
import optparse
import lasagne  as lgn
import nolearn.lasagne as nlln
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from utils import *

"""
For installing nolearn lib:

!pip uninstall Lasagne -y
!pip uninstall nolearn -y
!pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt
"""


if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.add_option("-m", "--matrix", dest="matrix_file", metavar="FILE",
                      help="filename with sparse feature matrix")
    parser.add_option("-t", "--targets", dest="targets_file", metavar="FILE",
                      help="filename with corresponding targets")
    parser.add_option("-s", "--split", dest="split", type=float, default=0.3,
                      help="Split ratio for constructing train/test split")
    parser.add_option("--balanced", dest="balanced", action="store_true", default=False,
                      help="whether the script should balance classes' distribution.")
    (options, _) = parser.parse_args()

    if options.matrix_file is None:
        raise RuntimeError("Specify the input filename with matrix, -m <filename>.")
    if options.targets_file is None:
        raise RuntimeError("Specify the input filename with targets, -t <filename>.")

    matrix_train, matrix_test,  targets_train, targets_test = construct_baseline_datasets((options.matrix_file,
                                                                                           options.targets_file),
                                                                                          split_ratio=options.split,
                                                                                          balanced=options.balanced)

    nnet = nlln.NeuralNet(
            layers=[ # two layers: one input and one output
                    ('input', lgn.layers.InputLayer),
                    ('output', lgn.layers.DenseLayer),
                    ],
            # layer parameters:
            input_shape=(None, matrix_train.shape[1]),
            output_nonlinearity=lgn.nonlinearities.softmax,
            output_num_units=2,  # 2 target values
            output_b = None, # no bias

            # optimization method:
            update=lgn.updates.nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            max_epochs=5,  # we want to train this many epochs
            verbose=1)

    # training

    nnet.fit(matrix_train.todense(), targets_train.astype(np.int32))

    # results

    print nnet.score(matrix_test.todense(), targets_test.astype(np.int32))

    expected = targets_test
    predicted = nnet.predict(matrix_test.todense())

    print "Classification report for classifier %s:\n%s\n" % (nnet, classification_report(expected, predicted))
    print "Confusion matrix:\n%s" % confusion_matrix(expected, predicted)

    nn_res = predictions_and_stats(nnet, matrix_test.todense(), targets_test)
    roc(nn_res[1], nn_res[2], nn_res[3], "Simple NN on Processed Matrix")
    plt.show()
