import csv
from collections import Counter
from typing import List, NamedTuple
from search_engine import Article
from sklearn.cluster import KMeans
import search_engine as engine
import sys
import nltk
import re
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from nltk.corpus import stopwords
from gensim.summarization.summarizer import summarize 

csv.field_size_limit(sys.maxsize)
# nltk.download('stopwords')
stop_words = stopwords.words('english')


def clean_text(text: str) -> str:
    ''''''
    clean_text = re.sub(r'[’”“]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


def get_text_sentences(text: str) -> List[str]:
    '''Cleanes text and splits into sentences with nltk

    Args:
        text: text to be splitted into sentences
    
    Returns:
        list of sentences
    '''
    new_text = clean_text(text)
    sentences = nltk.sent_tokenize(new_text)
    return sentences


def preprocess(text: str, remove_stop: bool=True) -> str:
    '''Preprocess text and possibly remove stopwords
    
    Args:
        text: text to be preprocessed
        remove_stop: whether to remove stop words
    
    Returns:
        preprocessed text
    '''
    return [t for t in engine.preprocess(text) if t not in stop_words]


def read_data(path: str = "articles50000.csv") -> List[Article]:
    '''Read data from original articles csv file
    
    Args:
        path: path to input csv file with second column
            containing title and third column - text

    Returns:
        List of read Articles
    '''
    result = []
    with open(path) as csvfile:
        articles = csv.reader(csvfile, delimiter=',')
        next(articles, None)
        for article in articles:
            result.append(Article(title=article[1], body=article[2]))
    
    return result


def naive_sum(doc: Article, query: str, summary_len: int) -> str:
    '''Implementaion of naive text summarization
    
    Idea is to take document data, preprocess it, divide into
    sentences, calculate score for each sentence based of tf
    of each word in it and multiply for tf of each word in 
    the query, and return top k sentences, for which
    sum of lengths is less or equal to `summary_len`

    Args:
        doc: text of the document
        query: input query
        summary_len: max amount of terms for output
    
    Returns:
        resulting summary (title + summary text)
    '''
    sentences = get_text_sentences(doc.body)
    
    # calculating number of term occurences in query and text
    q_tf = Counter(preprocess(query))
    tf = Counter(preprocess(doc.body))

    # normalizing tf on maximum tf
    max_freq = max(tf.values())
    for term in tf:
        tf[term] /= max_freq
    
    # calculating score for each sentence
    score_results = {}
    for sentence in sentences:
        # consider only short sentences
        if len(sentence.split(' ')) < 35:
            for term in preprocess(sentence):
                if sentence in score_results:
                    score_results[sentence] += tf[term] * q_tf[term]
                else:
                    score_results[sentence] = tf[term] * q_tf[term]
    
    result = [doc.title, '\n']
    cur_length = 0
    score_results = sorted(score_results.items(), key=lambda kv: kv[1], reverse=True)
    for sentence, _ in score_results:
        sent_len = len(sentence.split(' '))
        if cur_length + sent_len <= summary_len:
            result.append(sentence + ' ')
            cur_length += sent_len
        else:
            break
    
    return ''.join(result)


def build_graph(text: str, sentences: List[str], eps: float = 0.1) -> nx.Graph:
    '''TODO: add docstring'''
    n = len(sentences)
    tf = Counter(preprocess(text))

    idf = Counter()
    for s in sentences:
        for term in set(preprocess(s)):
            idf[term] += 1

    for term in idf:
        idf[term] = np.log10(n / (1 + idf[term]))
    
    V = np.zeros(shape=(n, len(tf)), dtype='float64')
    for i in range(len(sentences)):
        s_terms = preprocess(sentences[i])
        for j in range(len(tf)):
            term = list(tf.keys())[j]
            V[i, j] = tf[term] * idf[term] if term in s_terms else 0.0

    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            tf_idf_cos = np.sum(np.multiply(V[i], V[j]))
            tf_idf_cos /= np.sqrt(np.sum(V[i]**2)) * np.sqrt(np.sum(V[j]**2))
            if tf_idf_cos > eps:
                G.add_edge(i, j, weight=tf_idf_cos)
                G.add_edge(j, i, weight=tf_idf_cos)

    return G


def graph_sum(doc: Article, query: str, summary_len: int) -> str:
    '''Implementation of graph-based document summary algorithm

    The main idea is to build sentence graph with the usage of
    vector representation and tf-idf-cosine similarity to
    make connections between sentences. Afterwards apply
    summarization via K-means clustering of sentences and 
    returning sentences with largest degree from each cluster.

    This techniques is based on 
    http://tcci.ccf.org.cn/conference/2018/papers/SW1.pdf

    Args:
        doc: text of the document
        query: input query
        summary_len: max amount of terms for output
    
    Returns:
        resulting summary (title + summary text)
    '''
    result = [doc.title, '\n']
    thresh = 0.1

    sentences = get_text_sentences(doc.body)

    graph = build_graph(doc.body, sentences, thresh)
    node2vec = Node2Vec(graph, dimensions=20, num_walks=10, quiet=True, p=1)
    model = node2vec.fit()
    wvects = np.array([model.wv[str(i)] for i in range(len(sentences))])

    kmeans = KMeans(n_clusters=5)
    clusters = np.array(kmeans.fit_predict(wvects))

    sent_ids = []
    for i in range(5):
        nodes = np.argwhere(clusters == i)
        max_degree, _id = -1, -1
        for j in range(len(nodes)):
            if graph.degree(nodes[j][0]) > max_degree:
                _id = nodes[j][0]
                max_degree = graph.degree(nodes[j][0])
        sent_ids.append(_id)

    for i in sorted(sent_ids):
        result.append(sentences[i] + ' ')
            
    return ''.join(result)


def text_rank(doc: Article, query: str, summary_len: int) -> str:
    '''Call to TextRank implementation of gensim module

    Was selected for comparison with other implemented methods.

    Args:
        doc: text of the document
        query: input query
        summary_len: max amount of terms for output
    
    Returns:
        resulting summary (title + summary text)
    '''
    summary = summarize(clean_text(doc.body), word_count=summary_len)
    return doc.title + '\n' + summary


# TODO: redo `summary_len` into `sentence_cnt`
def compare_doc_sum(doc: Article, query: str, summary_len: int = 50):
    '''Funnction launches all summarization methods one-by-one

    Args:
        doc: text of the document
        query: input query
        summary_len: max amount of terms for output
    '''
    sum_methods = [naive_sum, graph_sum, text_rank]
    print('---------------------------------------')
    for method in sum_methods:
        print(f"Document summary for {method.__name__}")
        print(method(doc, query, summary_len))
        print('---------------------------------------')


def launch():
    data_path = 'data.nosync/articles50000.csv'
    save_dir = 'engine_data/'
    save_paths = {
        'index': f'{save_dir}index.p',
        'lengths': f'{save_dir}doc_lengths.p',
        'docs': f'{save_dir}documents.p'
    }
    
    if not engine.index_exists(paths=save_paths):
        print("* Building index... *")
        articles = read_data(data_path)
        engine.build_index(docs=articles, paths=save_paths, dir=save_dir)
        print("* Index was built successfully! *")
    else:
        print("* Loading index... *")
        engine.load_index(paths=save_paths)
        print("* Index was loaded successfully! *")
    
    q = "Macbook pro"
    docs = engine.answer_query(q, 2)
    print(docs[0])
    compare_doc_sum(docs[0], q, 100)

if __name__ == '__main__':
    launch()