import sys
import operator

class PCFG(object):
    def __init__(self,train_file, model_file, parse_file):
        self.train_file = train_file
        self.model_file = model_file
        self.parse_file = parse_file
        self.grammar = {}
        self.grammar_prob = {}
        self.start_symbol = []
    def load_data(self):
        operator_stack, tag_stack = [],[]
        with open(self.train_file, 'r') as fd:
            lines = fd.readlines()
            for line in lines:
                tmp = []
                word = []
                pair_flag = False
                for l in list(line.strip()):
                    if l == '(':
                        if word:
                           tag_stack.append(''.join(word))
                           word = []
                        operator_stack.append(l)
                    elif l == ')':
                        if word:
                           tag_stack.append(''.join(word))
                           word = []
                        if pair_flag:
                            val = tag_stack[-1].lower()
                            del tag_stack[-1]
                            key = tag_stack[-1]
                            del tag_stack[-1]
                            if (key, val) not in self.grammar.keys():
                                self.grammar[(key, val)] = (1, 1)
                            else:
                                (k, v) = self.grammar[(key, val)]
                                k += 1
                                self.grammar[(key, val)] = (k, v)
                            tmp.append(key)
                            pair_flag = False
                        else:
                            key = tag_stack[-1]
                            del tag_stack[-1]
                            pair = (tmp[-2], tmp[-1]) 
                            if (key, pair) not in self.grammar.keys():
                                self.grammar[(key, pair)] = (1, 0)
                            else:
                                (k, v) = self.grammar[(key, pair)]
                                k += 1
                                self.grammar[(key, pair)] = (k, v)
                            del tmp[-2:] 
                            tmp.append(key)
                        del operator_stack[-1]
                    elif l == ' ':
                        pair_flag = True
                        tag_stack.append(''.join(word))
                        word = []
                    else:
                        word.append(l)
    def train(self):
        if not self.grammar:
            self.load_data()
        start_symbol_freq = {}
        for (key, val) in self.grammar.iteritems():
            count = val[0]
            if key[0] in start_symbol_freq.keys():
                start_symbol_freq[key[0]]['count'] += count
                start_symbol_freq[key[0]]['values'].append( key[1] )
            else:
                start_symbol_freq[key[0]] = {}
                start_symbol_freq[key[0]]['values'] = [ key[1] ]
                start_symbol_freq[key[0]]['count'] = count 
        self.start_symbol = start_symbol_freq
        for key, val in self.grammar.iteritems():
            self.grammar_prob[key] = val[0]*1./start_symbol_freq[key[0]]['count']
        lines = []
        for pair in self.grammar_prob.keys():
            if type(pair[1]) is not tuple:
                lines.append('%s # %s # %.6f'%(pair[0], pair[1], self.grammar_prob[pair]))
            else:
                lines.append('%s # %s %s # %.6f'%(pair[0], pair[1][0], pair[1][1], self.grammar_prob[pair]))
        lines.sort()
        with open(self.model_file, 'w') as fd:
            fd.write('\n'.join(lines))

    def most_likely_parse(self,sentence):
        tokens = [ token.lower() for token in sentence.split(' ')]
        lent = len(tokens)
        start_symbol_list = [ key for key in self.start_symbol.keys() ]
        lens = len(start_symbol_list)
        sigma = [ [ [0.0 for j in xrange(lent) ] for i in xrange(lent) ] for k in xrange(lens)]
        path = [ [ [tuple() for j in xrange(lent) ] for i in xrange(lent) ] for k in xrange(lens)]
        for i, word in enumerate(tokens):
            for key, val in self.grammar_prob.iteritems():
                if type(key[1]) is not tuple and key[1] == word:
                    pos = start_symbol_list.index(key[0])
                    sigma[pos][i][i] = val   
                    path[pos][i][i] = ( tuple([word]), i )
        for l in xrange(1, lent):
            for i in xrange(lent - l):
                j = i + l 
                for k in xrange(lens):
                    max_val , max_index, max_rule = 0, 0, ''
                    for val in self.start_symbol[start_symbol_list[k]]['values']:
                        if type(val) is not tuple:
                            continue
                        index_l = start_symbol_list.index(val[0])
                        index_r = start_symbol_list.index(val[1])
                        _list = [sigma[index_l][i][s]*sigma[index_r][s+1][j]*self.grammar_prob[(start_symbol_list[k], val)] for s in xrange(i, j)]
                        s, new_val = max( enumerate(_list), key = operator.itemgetter(1))
                        if max_val < new_val:
                            max_index = s + i
                            max_val = new_val
                            max_rule = val
                    if max_rule != '':
                        sigma[k][i][j] = max_val
                        path[k][i][j] = ( max_rule, max_index)
        start, end , syb = 0, lent - 1, 'S'
        right_stack, output_string = [], ''
        count = []
        while syb in start_symbol_list  or len(right_stack) > 0:
            val = path[start_symbol_list.index(syb)][start][end]
            output_string += '(%s'%syb
            count.append(0)
            syb = val[0][0]
            if syb not in start_symbol_list:
                output_string += ' %s)'%syb
                del count[-1]
                count[-1] += 1
                while len(count) > 0 and count[-1] == 2:
                    output_string += ')'
                    del count[-1]
                    if len(count) > 0:
                        count[-1] += 1
                if len(right_stack) > 0:
                    (start, end, syb) = right_stack[-1]
                    del right_stack[-1]
                continue
            right_stack.append((val[1]+1, end, val[0][1]))
            end = val[1]
        return output_string, sigma[start_symbol_list.index('S')][0][lent-1]

    def logprob(self, sentence, parser_result, prob):
        tokens = [ token.lower() for token in sentence.split(' ')]
        start_symbol_list = [ key for key in self.start_symbol.keys() ]
        lent = len(tokens)
        lens = len(start_symbol_list) 
        beta = [ [ [0.0 for j in xrange(lent) ] for i in xrange(lent) ] for k in xrange(lens)]
        alpha = [ [ [0.0 for j in xrange(lent) ] for i in xrange(lent) ] for k in xrange(lens)]
        self.inner(tokens, start_symbol_list, beta)
        self.outer(tokens, start_symbol_list, alpha, beta)
        lines = []
        for i, syb in enumerate(start_symbol_list):
            for p in xrange(lent):
                for q in xrange(lent):
                    if beta[i][p][q] != 0 and alpha[i][p][q] != 0:
                        lines.append('%s # %d # %d # %.6f # %.6f'%(start_symbol_list[i], p, q, beta[i][p][q], alpha[i][p][q]))
        lines.sort()
        with open(self.parse_file, 'w') as fd:
            fd.write('%s\n%s\n'%(parse_result, prob))
            fd.write('\n'.join(lines))



    def inner(self, tokens, start_symbol_list, beta):
        for i, word in enumerate(tokens):
            for key, val in self.grammar_prob.iteritems():
                if type(key[1]) is not tuple and key[1] == word:
                    pos = start_symbol_list.index(key[0])
                    beta[pos][i][i] = val
        lent = len(tokens)
        for l in xrange(1, lent ):
            for i in xrange(lent - l ):
                j = i + l 
                for k in xrange(len(start_symbol_list)):
                    for val in self.start_symbol[start_symbol_list[k]]['values']:
                        if type(val) is not tuple:
                            continue
                        index_l = start_symbol_list.index(val[0])
                        index_r = start_symbol_list.index(val[1])
                        _list = [beta[index_l][i][s]*beta[index_r][s+1][j]*self.grammar_prob[(start_symbol_list[k], val)] for s in xrange(i, j)]
                        beta[k][i][j] += sum(_list)

    def outer(self, tokens, start_symbol_list, alpha, beta):
        alpha[start_symbol_list.index('S')][0][len(tokens)-1] = 1.
        lent = len(tokens)
        for l in reversed(xrange(lent - 1)):
            for i in xrange(0, lent - l ):
                j = i + l 
                for k in xrange(len(start_symbol_list)):
                    for val in self.start_symbol[start_symbol_list[k]]['values']:
                        if type(val) is not tuple:
                            continue
                        index_l = start_symbol_list.index(val[0])
                        index_r = start_symbol_list.index(val[1])
                        alpha[index_l][i][j] += sum([alpha[k][i][s]*self.grammar_prob[(start_symbol_list[k], val)]*beta[start_symbol_list.index(val[1])][j+1][s] for s in xrange(j+1, lent)])
                        alpha[index_r][i][j] += sum([alpha[k][s][j]*self.grammar_prob[(start_symbol_list[k], val)]*beta[start_symbol_list.index(val[0])][s][i-1] for s in xrange(0, i)])

if __name__ == '__main__':
    model = PCFG('../data/input.txt', '../data/model.txt', '../data/parse.txt')
    model.train()
    sentence = 'A boy with a telescope saw a girl'
    parse_result , prob = model.most_likely_parse(sentence)
    model.logprob(sentence, parse_result, prob)
