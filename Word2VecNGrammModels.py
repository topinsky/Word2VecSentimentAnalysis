import optparse
import gensim
import json
import bs4


def clean_text(corpus):
    punctuation = """.,?!:;(){}[]/"""
    corpus = [bs4.BeautifulSoup(z).get_text(separator=" ") for z in corpus]
    corpus = [z.lower().replace('\n', '') for z in corpus]

    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


def construct_n_gram(text, n_gram_value=1 ):
    results = []
    word_count = len(text)
    if n_gram_value > word_count:
        return []
    for i in xrange(word_count):
        if i <= word_count - n_gram_value:
            results.append('_'.join(text[i:i+n_gram_value]))
    return results


def get_cleaned_text(filename, n_gram_value=1, idf_filter=False):
    data_source = open(filename, 'r')
    raw_data = []
    for line in data_source:
        raw_text = json.loads(line.strip())['text']
        text = clean_text(raw_text)
        text_by_n_gram = construct_n_gram(text, n_gram_value)
        raw_data.append(text_by_n_gram)
    if idf_filter:

    return corpus



def main():
    parser = optparse.OptionParser()
    parser.add_option("-f", "--file", dest="corpus", metavar="FILE",
                      help="filename with textual corpus for creating Word2Vec model")
    parser.add_option("-g", "--n_gram", dest="n_gram", type=int, default=1,
                      help="n value for definition n-gram to be used")
    parser.add_option("-o", "--output", dest="output", metavar="FILE",
                      help="filename for created model")
    parser.add_option("--idf", dest="idf", action="store_true", default=False,
                      help="whether the model should keep only high idf words")
    (options, _) = parser.parse_args()

    corpus = get_cleaned_text(options.corpus, options.n_gram, options.idf)





if __name__ == '__main__':
    main()