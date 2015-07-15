# encoding=utf8
__author__ = 'topinsky'
import gzip
import optparse
import gensim
import json
import bs4
import numpy
import pylab

import sys
reload(sys)
sys.setdefaultencoding('utf8')


def clean_text(text):
    punctuation = """.,?!:;(){}[]/"""
    text_ = bs4.BeautifulSoup(text).get_text(separator=" ")
    text_ = text_.lower().replace('\n', '')

    for c in punctuation:
        text_ = text_.replace(c, ' %s ' % c)
    text_ = text_.split()
    return text_


def construct_n_gram(text, n_gram_value=1 ):
    results = []
    word_count = len(text)
    if n_gram_value > word_count:
        return []
    for i in xrange(word_count):
        if i <= word_count - n_gram_value:
            results.append('_'.join(text[i:i+n_gram_value]))
    return results


def preprocess_text(filename, n_gram_value=1):
    if filename[-3:] == '.gz':
        data_source = gzip.open(filename, 'rb')
        output_name = filename+'.pp.gz'
        storage = gzip.open(output_name, 'wb')
    else:
        data_source = open(filename, 'r')
        output_name = filename+'.pp'
        storage = open(output_name,'w')
    for line in data_source:
        raw_text = json.loads(line.strip())['text']
        text = clean_text(raw_text)
        text_by_n_gram = construct_n_gram(text, n_gram_value)
        storage.write(' '.join(text_by_n_gram) + '\n')
    storage.close()
    data_source.close()
    return output_name

def filter_text(filename, verbose=False):
    texts = gensim.models.word2vec.LineSentence(filename)
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tf_idf_model = gensim.models.TfidfModel(corpus, dictionary=dictionary, normalize=False)
    idfs = numpy.array(sorted(tf_idf_model.idfs.items(), key=lambda x: x[1]))
    if verbose:
        pylab.plot([idf[1] for idf in idfs])
        pylab.savefig("idf_cdf.png")
    total_idf = idfs.max(axis=0)[1]
    cum_idfs = 0
    bad_ids = []
    for elem in idfs:
        cum_idfs += elem[1]
        if cum_idfs / total_idf < 0.05:
            bad_ids.append(elem)
        else:
            break
    id2token = dict(dictionary.items())
    if verbose:
        print "WARNING:"
        print "The following tokens are going to be filtered out (cut-off 0.05 * Max IDF). Max IDF:", total_idf
        print repr([(id2token[bad_id[0]], bad_id[1])for bad_id in bad_ids])
    dictionary.filter_tokens(map(lambda x: x[0], bad_ids))
    filtered_tokens = set(dictionary.token2id.keys())
    if filename[-3:] == '.gz':
        output_name = filename[:-3] + '.fo.gz'
        storage = gzip.open(output_name, 'wb')
    else:
        output_name = filename + '.fo'
        storage = open(output_name, 'w')
    for text in texts:
        current_text = []
        for token in text:
            if token in filtered_tokens:
                current_text.append(token)
        storage.write(' '.join(current_text)+'\n')
    storage.close()
    return output_name


def construct_word2vec(filename, model='skip-gram'):
    texts = gensim.models.word2vec.LineSentence(filename)
    sg = 1 if model == 'skip-gram' else 0
    w2v = gensim.models.Word2Vec(texts, size=300, sg=sg)
    return w2v


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
    if options.output is None:
        raise RuntimeError("Specify the output filename, -o <filename>.")
    print "Starting..."
    corpus_file_ = preprocess_text(options.corpus, options.n_gram)
    print "Preprocessed..."
    if options.idf:
        corpus_file_ = filter_text(corpus_file_, options.verbose)
        print "Filtered..."
    word2vec_model = construct_word2vec(corpus_file_, options.model)
    print "W2V constructed..."
    word2vec_model.save_word2vec_format(options.output, binary=True)
    print "Done."

if __name__ == '__main__':
    main()
