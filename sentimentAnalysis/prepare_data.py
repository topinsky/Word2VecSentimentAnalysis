import optparse

from utils import *


def main():
    parser = optparse.OptionParser()
    parser.add_option("-f", "--file", dest="corpus", metavar="FILE",
                      help="filename with textual corpus for creating Word2Vec model")
    parser.add_option("-g", "--n_gram", dest="n_gram", type=int, default=1,
                      help="n value for definition n-gram to be used")
    parser.add_option("-o", "--output", dest="output", metavar="FILE",
                      help="filename for processed data")
    (options, _) = parser.parse_args()

    ifile = open(options.corpus, 'r')
    ofile = open(options.output, 'w')
    for line in ifile:
        data = json.loads(line.strip())
        data['text'] = construct_n_gram(clean_text(data['text']), options.ngram)
        data['class'] = int(data['stars'] > 3)
        ofile.write(json.dumps(data)+'\n')
    ofile.close()
    ifile.close()

if __name__ == '__main__':
    main()
