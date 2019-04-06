import nltk
import pickle
from collections import Counter
import math
import heapq
import os

index = {}
doc_lengths = {}
documents = {}


def tokenize(text):
    return nltk.word_tokenize(text)


def preprocess(text):
    tokenized = tokenize(text.lower())
    return [w for w in tokenized if w.isalpha()]


def build_index(docs: list, paths: dict):
    global index, doc_lengths, documents

    for doc_id, doc in enumerate(docs):
        full_doc = ''
        if (doc.title == '') or (doc.body == ''):
            full_doc = doc.title + doc.body
        else:
            full_doc = doc.title + '\n' + doc.body
        
        documents[doc_id] = full_doc

        doc_terms = preprocess(full_doc)
        doc_lengths[doc_id] = len(doc_terms)

        tf = Counter()
        for term in doc_terms:
            tf[term] += 1
        
        for term in tf:
            if term in index:
                index[term][0] += 1
            else:
                index[term] = [1]
            index[term].append((doc_id, tf[term]))

    with open(paths['index'], 'wb') as dump_file:
        pickle.dump(index, dump_file)
    
    with open(paths['lengths'], 'wb') as dump_file:
        pickle.dump(doc_lengths, dump_file)

    with open(paths['docs'], 'wb') as dump_file:
        pickle.dump(documents, dump_file)


def okapi_scoring(query, doc_lengths, index, k1=1.2, b=0.75):
    """
    Computes scores for all documents containing any of query terms
    according to the Okapi BM25 ranking function, refer to wikipedia,
    but calculate IDF as described in chapter 6, using 10 as a base of log

    :param query: dictionary - term:frequency
    :return: dictionary of scores - doc_id:score
    """
    scores = Counter()
    avgdl = sum(doc_lengths.values()) / len(doc_lengths)
    for term in query:
        if term in index:
            idf = math.log10(len(doc_lengths) / (len(index[term]) - 1))
            for i in range(1, len(index[term])):
                doc_id, doc_freq = index[term][i]
                nominator = doc_freq * (k1 + 1)
                denominator = (doc_freq + k1 * (1 - b + b * doc_lengths[doc_id] / avgdl))
                scores[doc_id] += idf * nominator / denominator
    
    return dict(scores)


def answer_query(raw_query, top_k):
    query = preprocess(raw_query)
    query = Counter(query)
    scores = okapi_scoring(query, doc_lengths, index)
    h = []
    for doc_id in scores.keys():
        neg_score = -scores[doc_id]
        heapq.heappush(h, (neg_score, doc_id))

    top_k_result = []
    top_k = min(top_k, len(h))
    for _ in range(top_k):
        best_so_far = heapq.heappop(h)
        article = documents[best_so_far[1]]
        top_k_result.append(article)

    return top_k_result


def index_exists(paths: dict):
    return os.path.isfile(paths['index'])


def load_index(paths: dict):
    global index, doc_lengths, documents

    with open(paths['index'], 'rb') as fp:
        index = pickle.load(fp)
    
    with open(paths['lengths'], 'rb') as fp:
        doc_lengths = pickle.load(fp)

    with open(paths['docs'], 'rb') as fp:
        documents = pickle.load(fp)