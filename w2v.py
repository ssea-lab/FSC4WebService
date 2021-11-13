'''
pre-train the word embeddings on the Web service dataset
'''

import json
import joblib
import numpy as np
import os
from gensim.models import Word2Vec

DATA_ROOT = './data/'
CACHE_ROOT = './cache/'
PRE_TRAIN_MODEL_NAME="wiki.en.vec"
EMB_DIM = 300
dataset = 'pw' # ['pw', 'aws']
assert dataset in ['pw', 'aws']
MODEL_NAME = '%s_word2vec.txt' %dataset

# clear cache
if os.path.exists(os.path.join(CACHE_ROOT,MODEL_NAME + '.pt')):
    os.remove(os.path.join(CACHE_ROOT,MODEL_NAME + '.pt'))

index2label = joblib.load(os.path.join(DATA_ROOT,'%s_index2label.pkl'%dataset))
sents = []
counter = {}
with open(os.path.join(DATA_ROOT,'%s.json'%dataset),'r', errors='ignore', encoding='utf8') as fr:
    for line in fr:
        row = json.loads(line)
        sents.append(row['text'])
        for word in set(row['text']):
            counter[word] = counter.get(word,0) + 1

for label_tokens in index2label.values():
    sents.append(label_tokens)

w2v_model = Word2Vec(size=EMB_DIM, sg=1, min_count=1, window=9)
w2v_model.build_vocab(sents)

w2v_model.intersect_word2vec_format(os.path.join(CACHE_ROOT,PRE_TRAIN_MODEL_NAME), binary=False, lockf=1.0) 
w2v_model.train(sents, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
w2v_model.wv.save_word2vec_format(os.path.join(CACHE_ROOT, MODEL_NAME))

# generate embedding vectors for categories
vectors = w2v_model.wv
index2vector = {}
for index,label_tokens in index2label.items():
    vector = np.zeros((EMB_DIM,))
    l = 0
    idf_sum = 0
    for token in label_tokens:
        if token in vectors:
            vector += vectors[token] 
            l += 1
        else:
            print(token)
    if l > 0:
        vector /= l    
    index2vector[index] = vector
joblib.dump(index2vector,os.path.join(DATA_ROOT,'%s_index2vector.pkl'%dataset))

