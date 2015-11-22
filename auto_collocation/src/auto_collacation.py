# -*- coding: utf-8 -*-
import pdb
import re
import codecs
import nltk
from os import walk
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np

''' \u4e00-\u9fff simplified chinese; 
    \uff10-\uff19 gbk numbers;
    \uff0c \uff01 gbk ',', '!';
'''


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

def frequency(word_tagged_set, wordpair_set, combinations, outpath):
    common_set = set(['U','P','R','M', 'Q', 'T'])
    wordpair_fd = nltk.FreqDist( (a[0],b[0], a[1], b[1]) for (a, b) in wordpair_set) 
    tag_combination_fd = nltk.FreqDist( (word, tag) for (word, tag) in combinations)
    # Naively rank all biagram pairs
    naive_counts = [ ' '.join(list(collocation)+['%d'%count]) for (collocation ,count) in  wordpair_fd.most_common(100)]
    # Filter those collocation with less sense, quantity collocations
    filter_counts = [ ' '.join(list(collocation)+['%d'%count]) for (collocation ,count) in  wordpair_fd.most_common() if (( collocation[2] not in common_set) and ( collocation[3] not in common_set)) ]
    # count bigram pair frequency by tag type
    wordpair_conditional_fd = nltk.ConditionalFreqDist( (' '.join([ a[1],b[1] ]), ' '.join([ a[0],b[0] ])) for (a, b) in wordpair_set )
    category_rank = sorted([ (wordpair_conditional_fd[index].N(), index) for index in wordpair_conditional_fd.conditions() ], reverse=True)
    # Get most frequent meaningful collocation by type
    categoryList = [ ':\n'.join([ st, '\n'.join([ '%s %d'%(word, count)  for (word, count) in wordpair_conditional_fd[st].most_common(10) ]) ]) for (i, st) in category_rank[0:10]  ]
    with open('%s.col_by_tag'%outpath, 'w') as fd:
        fd.write('\n'.join(categoryList).encode('utf-8'))
    with open('%s.category_rank'%outpath, 'w') as fd:
        fd.write('\n'.join( [ '%d %s'%(i, st) for (i, st) in category_rank]))
    with open('%s.rm_quant_tags_rank'%outpath, 'w') as fd:
        fd.write('\n'.join(filter_counts).encode('utf-8'))
    with open('%s.unfiltered_rank'%outpath, 'w') as fd:
        fd.write('\n'.join(naive_counts).encode('utf-8'))

def mutual_information(word_tagged_set, wordpair_set, combinations, outpath):
    exclude_set = set(['P','M','QG'])
    pair_by_word = nltk.FreqDist( ( a[0],b[0] ) for (a, b) in wordpair_set if ((a[1] not in exclude_set) and (b[1] not in exclude_set))) 
    freqDict = nltk.ConditionalFreqDist((word , 1)  for (word, tag) in word_tagged_set )
    pair_counts = len(wordpair_set)
    word_counts = len(word_tagged_set)
    # count mutual information for all bigram pairs
    mutual_values = [ ( math.log((count*1.0/pair_counts)/(count1*count2)*(word_counts**2)),[ word1,word2, count1,count2, count] ) for ((word1, word2), count) in pair_by_word.most_common() for (t1, count1) in freqDict[word1].most_common() for (t2, count2) in freqDict[word2].most_common()]
    mutual_values_descend = sorted(mutual_values, reverse = True)
    with open('%s.tag_filtered'%outpath, 'w') as fd:
        fd.write("\t".join([ "互信息", "词组", "C(w1)","C(w2)","C(w1,w2)"]))
        fd.write("\n")
        fd.write("\n".join([ "\t".join( ["%.8lf"%val]+ [word1.encode("utf-8"), word2.encode("utf-8"), "%d"%count1, "%d"%count2, "%d"%count]) for (val, [ word1, word2, count1, count2, count ]) in mutual_values_descend]))
    # retrieve mutual information for bigram pairs of frequency 20
    frequency_twenty_mutuals = [(val, [ word1, word2, count1, count2, count ]) for (val, [ word1, word2, count1, count2, count ]) in mutual_values if count == 20]
    frequency_twenty_mutuals_descend = sorted(frequency_twenty_mutuals, reverse = True)    
    with open('%s.count_filtered'%outpath, 'w') as fd:
        fd.write("\t".join([ "互信息", "词组", "C(w1)","C(w2)","C(w1,w2)"]))
        fd.write("\n")
        fd.write("\n".join([ "\t".join( ["%.8lf"%val]+ [word1.encode("utf-8"), word2.encode("utf-8"), "%d"%count1, "%d"%count2, "%d"%count]) for (val, [ word1, word2, count1, count2, count ]) in frequency_twenty_mutuals_descend]))

if __name__ == '__main__':
    input_path = '../data/input'
    preprocessed_path = '../data/output/processed_data.pkl'
    frequency_path = '../data/output/frequency'
    mutual_path = '../data/output/mutual'
    #word_tagged_set, combinations, wordpair_set = load_data(input_path)
    print 'Write to module file %s'%preprocessed_path
    #saveToFile((word_tagged_set, combinations,  wordpair_set), preprocessed_path)
    print 'Load from module file %s'%preprocessed_path
    (word_tagged_set,combinations, wordpair_set) = loadFromFile(preprocessed_path)
    dist = nltk.FreqDist((words, tag) for (words, tag) in combinations)
    print '\n'.join([ words.encode("utf-8")+" "+ tag+" %d"%_  for ((words, tag),_) in dist.most_common()])
    #statistics(word_tagged_set, combinations)
    #frequency(word_tagged_set,wordpair_set, combinations, frequency_path)
    #mutual_information(word_tagged_set,wordpair_set, combinations, mutual_path)
    #print '\n\n\nPrint all bigram words...'
    #printBigramTupleList( wordpair_set )
