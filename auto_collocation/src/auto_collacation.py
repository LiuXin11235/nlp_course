import pdb
import re
import codecs
import nltk
from os import walk
import pickle
import matplotlib.pyplot as plt
import numpy as np

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
    for i, filename in enumerate(f):
        if i == 10:
            break
        with codecs.open(mypath+'/'+filename, 'r',encoding='gbk') as fd:
            tagged_article = []
            combination = []
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
def statistics(word_tagged_set, combinations):
    tag_total_fd = nltk.ConditionalFreqDist( (tag, word) for (word, tag) in word_tagged_set) 
    print combinations
    tag_combination_fd = nltk.ConditionalFreqDist( (tag, word) for (word, tag) in combinations)
    counts = [ ( tag_total_fd[index].N(), index) for index in tag_total_fd.conditions()]
    counts += [ (tag_total_fd[index].N(), index) for index in tag_combination_fd.conditions()]
    counts = sorted(counts)
    tags = [('%s'%index).encode('utf-8') for (_, index) in counts]
    fig, ax = plt.subplots()
    width = 0.9
    ind = np.arange(len(tags))
    rects1 = ax.bar(ind , [ _ for ( _, index) in counts] , width, color='r',label='Number of Words in Each Tag')
    plt.xticks(ind + width, rotation=45)
    ax.set_xticklabels(tags)
    plt.title('Word Distribution')
    plt.xlabel('Tags')
    plt.ylabel('Number of Words')
    plt.show()

def frequency(word_tagged_set, wordpair_set, combinations):
    #tags = set([tag for (word, tag ) in word_tagged_set]+ [tag for (word, tag) in combinations])
    common_set = set(['U','P','R','M', 'Q', 'T'])
    wordpair_fd = nltk.FreqDist( (a[0],b[0], a[1], b[1]) for (a, b) in wordpair_set) 
    tag_combination_fd = nltk.FreqDist( (word, tag) for (word, tag) in combinations)
    # Naively rank all biagram pairs
    naive_counts = [ ' '.join(list(collocation)+['%d'%count]) for (collocation ,count) in  wordpair_fd.most_common(100)]
    # Filter those collocation with less sense
    filter_counts = [ ' '.join(list(collocation)+['%d'%count]) for (collocation ,count) in  wordpair_fd.most_common() if (( collocation[2] not in common_set) and ( collocation[3] not in common_set)) ]
    # count bigram pair frequency by tag type
    wordpair_conditional_fd = nltk.ConditionalFreqDist( (' '.join([ a[1],b[1] ]), ' '.join([ a[0],b[0] ])) for (a, b) in wordpair_set )
    category_rank = sorted([ (wordpair_conditional_fd[index].N(), index) for index in wordpair_conditional_fd.conditions() ], reverse=True)
    # Get most frequent meaningful collocation by type
    categoryList = [ ':\n'.join([ st, '\n'.join([ '%s %d'%(word, count)  for (word, count) in wordpair_conditional_fd[st].most_common(10) ]) ]) for (i, st) in category_rank[0:10]  ]
    #print '\n'.join(categoryList).encode('utf-8')
    #print '\n'.join( [ '%d %s'%(i, st) for (i, st) in category_rank])
    #print '\n'.join(filter_counts).encode('utf-8')
     
if __name__ == '__main__':
    input_path = '../data/input'
    preprocessed_path = '../data/output/processed_data.pkl'
    #word_tagged_set, combinations, wordpair_set = load_data(input_path)
    print 'Write to module file %s'%preprocessed_path
    #saveToFile((word_tagged_set, combinations,  wordpair_set), preprocessed_path)
    print 'Load from module file %s'%preprocessed_path
    (word_tagged_set,combinations, wordpair_set) = loadFromFile(preprocessed_path)
    #statistics(word_tagged_set, combinations)
    frequency(word_tagged_set,wordpair_set, combinations)
    print '\n\n\nPrint all bigram words...'
    #printBigramTupleList( wordpair_set )
    #print wordset
