import csv
from collections import Counter
from typing import List, NamedTuple
from search_engine import Article
import search_engine as engine
import sys
import nltk
import re

csv.field_size_limit(sys.maxsize)


def read_data(path="articles1000.csv"):
    '''Read data from original articles csv file
    
    Args:
        path (str): path to input csv file with second column
            containing title and third column - text

    Returns:
        list[Article] - list of Article namedtuples
    '''
    result = []
    with open(path) as csvfile:
        articles = csv.reader(csvfile, delimiter=',')
        next(articles, None)
        for article in articles:
            result.append(Article(title=article[1], body=article[2]))
    
    return result


def doc_sum_1(doc: Article, query: str, summary_len: int) -> str:
    '''Implementaion of naive text summarization
    
    Idea is to take document data, preprocess it, divide into
    sentences, calculate score for each sentence based of tf
    of each word in it and multiply for tf of each word in 
    the query, and return top k sentences, for which
    sum of lengths is less or equal to `summary_len`

    Args:
        doc (str): text of the document
        query (str): input query
        summary_len (int): max amount of terms for output
    
    Returns:
        str: resulting summary (title + summary text)
    '''
    # doc_clear = doc.body#re.sub(r'\[[0-9]*\]', ' ', doc.body)
    doc_clear = re.sub(r'[’”]', ' ', doc.body)
    doc_clear = re.sub(r'\s+', ' ', doc_clear)
    sentences = nltk.sent_tokenize(doc_clear)
    
    # calculating number of term occurences in query and text
    q_tf = Counter(engine.preprocess(query))
    tf = Counter(engine.preprocess(query))

    # normalizing tf on maximum tf
    max_freq = max(tf.values())
    for term in tf:
        tf[term] /= max_freq
    
    # calculating score for each sentence
    score_results = {}
    for sentence in sentences:
        for term in engine.tokenize(sentence):
            # consider only short sentences
            if len(sentence.split(' ')) < 25:
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


def doc_sum_2(doc: Article, query: str, summary_len: int) -> str:
    pass

def doc_sum_3(doc: Article, query: str, summary_len: int) -> str:
    pass

def compare_doc_sum(doc: Article, query: str, summary_len: int = 50):
    sum_methods = [doc_sum_1, doc_sum_2, doc_sum_3]
    for method in sum_methods:
        print(f"Document summary for {method.__name__}")
        print(method(doc, query, summary_len))

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
    
    q = "Tesla model x"
    docs = engine.answer_query(q, 2)
    compare_doc_sum(docs[0], q, 100)

if __name__ == '__main__':
    launch()