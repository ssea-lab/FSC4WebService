'''
preprocess the original Web services 
'''

import re
import os
import json
import random
import joblib
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords


DATA_ROOT = './data/'
DATASETS =['pw']    
MIN_COUNT = 6
    
stop_words = set(stopwords.words('english'))
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
stop_words.update(english_punctuations)

special_words = {'3d':['3d'],'iot':['iot'],'elearning':['elearning'],'it':['it'],'ecommerce':['ecommerce']}

need_replace_words = {'Text/Captioning':'Captioning','Edit/Processing-Text':'Processing-Text',\
                    'Edit/Processing-Image':'Processing-Image','Names Entity Recognition - NER':'Names Entity Recognition',\
                    'Amazon SageMaker Ground Truth Services':'Ground Truth','Edit/Processing-Video':'Processing-Video',\
                    'Text/OCR':'OCR','ELT/ETL':'Extract-Transform-Load','Continuous Integration and Continuous Delivery':'Continuous Integration Delivery'}

def _replacer(text):
    replacement_patterns = [
        (r'won\'t', 'will not'),
        (r'can\'t', 'cannot'),
        (r'i\'m', 'i am'),
        (r'ain\'t', 'is not'),
        (r'(\w+)\'ll', r'\g<1> will'),
        (r'(\w+)n\'t', r'\g<1> not'),
        (r'(\w+)\'ve', r'\g<1> have'),
        (r'(\w+)\'s', r'\g<1> is'),
        (r'(\w+)\'re', r'\g<1> are'),
        (r'(\w+)\'d', r'\g<1> would')]
    patterns = [(re.compile(regex), repl) for (regex, repl) in replacement_patterns]
    s = text
    for (pattern, repl) in patterns:
        (s, _) = re.subn(pattern, repl, s)
    return s

        
def my_tokenize(seq):
    seq = re.sub('https?://\S* ', '', seq)
    seq = _replacer(seq)  
    for k,v in need_replace_words.items():
        seq = seq.replace(k,v)
    a = word_tokenize(seq)
    b = []
    for word in a:
        if word.lower() in special_words:
            b.extend(special_words[word.lower()])
        else:
            candidates = list(filter(lambda x : x.strip() != '',word.split('-')))
            for candidate in candidates:
                pre = ''
                for chr in candidate:
                    if not chr.isalpha():
                        if pre != '':
                            b.append(pre.lower())
                            pre = ''
                    else:
                        if chr.islower():
                            pre += chr
                        else:
                            if pre == '':
                                pre += chr
                            else:
                                if pre[-1].islower():
                                    b.append(pre.lower())
                                    pre = chr
                                else:
                                    pre += chr
                if pre != '':
                    b.append(pre.lower())
    b = [x for x in b if x not in stop_words and not x.isdigit()]
    return b

for dataset in DATASETS:
    assert dataset in ['pw','aws']
    print('Processing dataset: %s'%dataset)
    train_indices = set()
    test_indices = set()
    res = []
    test_count = 0
    test_label_count = 0
    train_label_count = 0
    index2label = {}
    label_word_tokens = []
    iterators = None
    if dataset == 'pw':
        MAX_COUNT_TEST = 30
        MAX_COUNT_TRAIN = 100 # the max service count for training set, just for saving running time
        apis = pd.read_csv(os.path.join(DATA_ROOT,'pw_services.csv'))
        apis.fillna('',inplace=True)
        apis = apis.dropna(subset=['Description'])
        apis['Categories'] = apis['Categories'].apply(lambda x : eval(x))
        index2line = {}
        label_index = {}
        for index,line in enumerate(apis.itertuples()):
            index2line[index] = line
            for label in line.Categories:
                if label == '':
                    continue
                if label not in label_index:
                    label_index[label] = []
                label_index[label].append(index)
        label_index = {k:v for k,v in label_index.items() if len(v) >= MIN_COUNT and len(v) <= MAX_COUNT_TRAIN}
        label_list = list(label_index.keys()) 
        for v in label_index.values():
            if len(v) <= MAX_COUNT_TEST:
                test_indices.update(v)
        iterators = label_index
    else:
        MAX_COUNT_TEST = 150
        MAX_COUNT_TRAIN = 1000
        apis = json.load(open(os.path.join(DATA_ROOT,'amazon_services.json'),'r',encoding='utf8'))
        label_id = {}
        id_item = {}
        for item in apis:
            id_item[item['Id']] = item
            for label in item['Categories']:
                if label == '':
                    continue
                if label not in label_id:
                    label_id[label] = []
                label_id[label].append(item['Id'])
        label_id = {k:v for k,v in label_id.items() if len(v) >= MIN_COUNT and len(v) <= MAX_COUNT_TRAIN}
        label_list = list(label_id.keys())
        for v in label_id.values():
            if len(v) <= MAX_COUNT_TEST:
                test_indices.update(v)
        iterators = label_id
    print('Label count: {}'.format(len(label_list)))
    
    label = 0
    for k,v in sorted(iterators.items(), key=lambda x: len(x[1])):
        label_tokens = my_tokenize(k)
        label_word_tokens.append(' '.join(label_tokens) + '--------' + k)
        index2label[label] = label_tokens   
        if len(v) <= MAX_COUNT_TEST:
            test_count += len(v)
            test_label_count += 1
        else:
            train_label_count += 1
            v = [x for x in v if x not in test_indices]
            train_indices.update(v)
        if dataset == 'pw':
            for desc_index in v:
                line = index2line[desc_index]
                flag = False
                tmp = {}
                text = my_tokenize(line.Description)
                tmp['name'] = line.Name
                tmp['text'] = text
                tmp['label'] = label
                tmp['raw'] = line.Description
                if len(text) > 0:
                    res.append(tmp)
        else:
            for id in v:
                item = id_item[id]
                flag = False
                tmp = {}
                text = my_tokenize(item['ShortDescription'])
                tmp['name'] = item['Title']
                tmp['text'] = text
                tmp['label'] = label
                tmp['raw'] = item['ShortDescription']
                if len(text) > 0:
                    res.append(tmp)
        label += 1

    print('common label count between train and test dataset: %d'%len(train_indices&test_indices))
    print('test label count: {}'.format(test_label_count))
    print('train label count: {}'.format(train_label_count))
    
    test_sub = res[:test_count]
    train_sub = res[test_count:]
    random.shuffle(test_sub)
    random.shuffle(train_sub)
    res = test_sub + train_sub

    with open(os.path.join(DATA_ROOT,'%s.json'%dataset),'w') as fw:
        for line in res:
            fw.write(json.dumps(line)+'\n')
    joblib.dump(index2label,os.path.join(DATA_ROOT,'%s_index2label.pkl'%dataset))
        
    with open(os.path.join(DATA_ROOT,'%s_label_tokens.txt'%dataset),'w') as fw:
        fw.write('\n'.join(label_word_tokens))    
    