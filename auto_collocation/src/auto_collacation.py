import pdb
import re
import codecs
import nltk
from os import walk
import pickle

''' \u4e00-\u9fff simplified chinese; 
    \uff10-\uff19 gbk numbers;
    \uff0c \uff01 gbk ',', '!';
'''
PATTERN=ur'[\u4e00-\u9fff\uff10-\uff19\uff0c\uff01\u3002]+'
SEPARATER=ur'[\uff0c\uff01\u3002]+'


def tagger(t, combination):
    patterns = [(r'(.*?)\]nt$', 'NT'),(r'(.*?)\]nz$', 'NZ'),(r'(.*?)\]ns$', 'NS'), 
            (r'(.*?)\]l$','L'), (r'(.*?)\]i$', 'I'), (r'^\[(.*?)$', 'START')]
    match_res = [ re.match(rule, t) if re.match(rule, t) else [] for (rule, ID) in patterns ]
    flag = 0
    tagged_article = []
    combinations = []
    for i in xrange(len(patterns)):
        if match_res[i]:
            flag = 1
            matched = nltk.tag.str2tuple(match_res[i].group(1))
            tagged_article.append(matched) 
            if i == 5:
                combination.append(matched[0])
            else:
                combination.append(matched[0])
                combinations.append((' '.join(combination), patterns[i][1]))
                combination = []
    if flag == 0:
        tagged_article.append(nltk.tag.str2tuple(t))
        if combination:
            combination.append(nltk.tag.str2tuple(t)[0])
    return tagged_article, combination, combinations

def printTupleList(list_):
    print ' '.join([a.encode('utf-8') for (a, b) in list_])

def printBigramTupleList(list_):
    print '\n'.join([' '.join( [a[0], b[0]] ).encode('utf-8') for (a, b) in list_])

def saveToFile(data, outputPath):
    output = open(outputPath, 'wb')
    pickle.dump(data, output, -1)
    output.close()

def loadFromFile(inputPath):
    _input = open(inputPath, 'rb')
    data = pickle.load(_input)
    _input.close()
    return data

def load_data(mypath):
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    word_tagged_set = []
    wordpair_set = []
    combinations = []
    combination = []
    tagged_article = []
    for i, filename in enumerate(f):
        with codecs.open(mypath+'/'+filename, 'r',encoding='gbk') as fd:
            print "process file %d"%i
            for line in fd:
                for t in line.split():
                    delta_tag,combination, delta_com = tagger(t, combination)
                    tagged_article += delta_tag
                    combinations += delta_com
            word_tagged_set += [ (word, tag) for (word, tag) in tagged_article if tag != 'W']
            wordpair_set += [ (a, b) for (a, b) in nltk.bigrams(tagged_article) if ( ( a[1] != 'W') and (b[1] != 'W') )]
    return word_tagged_set, combinations, wordpair_set


def process_data(data):
    ( word_tagged_set, wordpair_set ) = data
    print set([tag for (word, tag ) in word_tagged_set])
    tag_fd = nltk.FreqDist((word, tag) for (word, tag) in word_tagged_set if tag != 'W')
    wordpair_fd = nltk.FreqDist( (a[0], b[0]) for (a, b) in wordpair_set if ((a[1] != 'W') and (b[1] != 'W')))
    #print tag_fd.most_common(100)
    list_ = [  word[0].encode('utf-8') for ( word, _) in tag_fd.most_common(100)  ] 
    print ' '.join(list_)
    list_pair = [  ' '.join([ word[0].encode('utf-8'), word[1].encode('utf-8'), '%d'%_]) for ( word, _) in wordpair_fd.most_common(100)  ] 
    print '\n'.join(list_pair)
    #for (word1,word2), in wordpair_fd.most_common(100): 
     #   if word[1] != 'W':
      #      print  word[0].encode('utf-8'),word[1] , _
   # print [ for (word, count) in wordpair_fd.most_common()]
    ''' for line in fd:
                single_sentence = []
                for matched in re.findall(PATTERN,text):
                    if matched == re.match(SEPARATER, matched): 
                        single_sentence = []
        '''   # print text.encode('utf-8')
 #           for i in text.split(' ') if i]
  #          wordset.append(text.split(' ').)
    return word_tagged_set

if __name__ == '__main__':
    input_path = '../data/input'
    preprocessed_path = '../data/output/processed_data.pkl'
    word_tagged_set, combinations, wordpair_set = load_data(input_path)
    print 'Write to module file %s'%preprocessed_path
    saveToFile((word_tagged_set, combinations,  wordpair_set), preprocessed_path)
    print 'Load from module file %s'%preprocessed_path
    (word_tagged_set,combinations, wordpair_set) = loadFromFile(preprocessed_path)
    print '\n\n\nPrint all words...'
    #printTupleList(word_tagged_set)
    print '\n\n\nPrint all combinations words...'
    #printTupleList(combinations)
    print '\n\n\nPrint all bigram words...'
    #printBigramTupleList( wordpair_set )
    #print wordset
