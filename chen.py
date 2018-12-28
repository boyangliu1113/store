#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import csv
import json
import numpy as np
import gensim
from sklearn.feature_extraction.text import CountVectorizer

def clean_text(s):
    if type(s) != str:
        s = s.decode('utf-8')
    s = s.encode('ascii', errors='ignore')
    return s

def main(infile, outfile):
    print("Load Word2Vec")
    google_news_word2vec = './GoogleNews-vectors-negative300.bin'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(google_news_word2vec, binary=True)
    vectorizer = CountVectorizer(stop_words="english")
    print("Start Calculating")
    with open(infile, "r") as fin, open(outfile, "w") as fout:
        reader = csv.DictReader(fin)
        fields = ["title", "subtitle", "gp_name", "gp_category", "similarity"]
        writer = csv.DictWriter(fout, fields)
        writer.writeheader()
        for item in reader:
            ad_title = item["title"].strip()
            if not ad_title:
                ad_title = item["subtitle"].strip()
            gp_name = item["gp_name"]
            try:
                x = vectorizer.fit_transform([ad_title, gp_name])
            except Exception as e:
                print(e)
                continue
            x = x.toarray()
            words = vectorizer.get_feature_names()
            has_vec = [w in word2vec for w in words]
            merged = [False for i in range(len(words))]
            weights = []
            diff = []
            for i, word in enumerate(words):
                if merged[i]:
                    continue
                if has_vec[i]:
                    for j in range(i+1, len(words)):
                        if not merged[j] and has_vec[j] and word2vec.similarity(word, words[j]) > 0.5:
                            merged[j] = True
                            x[:, i] = x[:, i] + x[:, j]
                    weights.append(1)
                else:
                    if word.isdigit() or clean_text(word) != word:
                        weights.append(1)
                    else:
                        weights.append(4)
                diff.append(x[0, i] * x[1, i] == 0)
            min_words_cnt = np.min(np.sum(x, axis=1))
            if not diff or not min_words_cnt:
                item['similarity'] = '*'
            else:
                similarity = 1 - np.dot(np.array(diff, dtype=float), np.array(weights, dtype=float)) / np.sum(weights)
                similarity = min(1.0, similarity * (1 + np.log(min_words_cnt) / 2))
                item['similarity'] = similarity
            writer.writerow({k:item[k] for k in fields})

if __name__ == '__main__':
    infile = sys.argv[1] if len(sys.argv) > 1 else "adinfo_keywords.csv"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "adinfo_gp_sim.csv"
    main(infile, outfile)

