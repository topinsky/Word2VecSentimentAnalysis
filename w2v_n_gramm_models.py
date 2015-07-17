# encoding=utf8
__author__ = 'topinsky'
import os
import optparse

from utils import *


def main():
    parser = optparse.OptionParser()
    parser.add_option("-f", "--file", dest="corpus", metavar="FILE",
                      help="filename with textual corpus for creating Word2Vec model")
    parser.add_option("-g", "--n_gram", dest="n_gram", type=int, default=1,
                      help="n value for definition n-gram to be used")
    parser.add_option("-m", "--model", dest="model", type='choice', choices=['skip-gram', 'cbow'], default='skip-gram',
                      help="Specify the choice of Word2Vec model: skip-gram or cbow")
    parser.add_option("-o", "--output", dest="output", metavar="FILE",
                      help="filename for created model")
    parser.add_option("--idf", dest="idf", action="store_true", default=False,
                      help="whether the model should keep only high idf words")
    parser.add_option("--verbose", dest="verbose", action="store_true", default=False,
                      help="whether the script should output auxiliary information.")
    (options, _) = parser.parse_args()

    if options.corpus is None:
        raise RuntimeError("Specify the input filename, -f <filename>.")

    print "Starting..."
    if not os.path.isfile(options.corpus + ('.pp%d' % n_gram_value)) \
            and not os.path.isfile(options.corpus + ('.pp%d.gz' % n_gram_value)):
        corpus_file_ = preprocess_text(options.corpus, options.n_gram)
    print "Preprocessed..."
    if options.idf:
        if not os.path.isfile(options.corpus + ('.pp%d.fo' % n_gram_value)) \
                and not os.path.isfile(options.corpus + ('.pp%d.fo.gz' % n_gram_value)):
            corpus_file_ = filter_text(corpus_file_, options.verbose)
        print "Filtered..."
    if options.output is None:
        print "W2V skipped..."
    else:
        word2vec_model = construct_word2vec(corpus_file_, options.model)
        print "W2V constructed..."
        word2vec_model.save_word2vec_format(options.output, binary=True)
    print "Done."


if __name__ == '__main__':
    main()
