import json
import re
import numpy as np
import search_engine as engine
from search_engine import Article

def read_data(path: str):
    """
    Helper function, parses Cranfield data. Used for tests. Use it to evaluate your own search engine
    :param path: original data path
    :return: dictionaries - documents, queries, relevance
    relevance comes in form of tuples - query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    """
    documents = {}
    queries = {}
    relevance = {}
    for doc in json.load(open(path + 'cranfield_data.json')):
        # if doc['body'] != '':
        title = re.sub(r'\s+', ' ', doc['title'])
        body = re.sub(r'\s+', ' ', doc['body'][len(doc['title']):])
        documents[doc['id']] = Article(title=title, body=body)
    
    for query in json.load(open(path + 'cran.qry.json')):
        queries[query['query number']] = query['query']
    for rel in json.load(open(path + 'cranqrel.json')):
        query_id = int(rel['query_num'])
        doc_id = int(rel['id'])
        if query_id in relevance:
            relevance[query_id].append((doc_id, rel['position']))
        else:
            relevance[query_id] = [(doc_id, rel['position'])]
    return documents, queries, relevance


def combine_score_and_docs(docs, scores):
    result = []
    for doc in docs:
        score = 0
        for sc in scores:
            if sc[0] == doc:
                score = 5 - sc[1]
                break

        result.append((doc, score))
    
    return result


def DCG(doc2relevance, top_k):
    dsg = 0.0
    for i, doc2relev in enumerate(doc2relevance[:top_k]):
        dsg += (2**doc2relev[1] - 1) / np.log2(2 + i)

    return dsg


def NDCG(top_k_results, relevance, top_k):
    '''
    Computes NDCG score for search results

    # :param top_k_results: list of lists of ranked results for each query [[doc_id1, doc_id2,...], ...]
    #                       the i-th result corresponds to (i+1)-th query_id. There may be less than top_k
    #                       results returned for a query, but never more.
    # :param relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    # :param top_k: (max) number of results retrieved for each query, use this value to find normalization
    #               factor for each query
    # :return: NDCG score
    '''
    ndcg_score = 0.0
    for j, scores in relevance.items():
        doc2relevance = combine_score_and_docs(top_k_results[j - 1], scores)
        dcg_score = DCG(doc2relevance, top_k)
        upd_score = list(map(lambda x: (x[0], 5 - x[1]), scores))
        idcg_score = DCG(sorted(upd_score, key=lambda _: _[1], reverse=True), top_k)
        ndcg_score += 0.0 if idcg_score == 0 else dcg_score / idcg_score
        
    ndcg_score /= len(relevance)

    return ndcg_score


def local_method1():
    pass


def global_method1():
    pass


def launch():
    data_path = 'data.nosync/cranfield/'
    save_dir = 'engine_data/query_exp/'
    save_paths = {
        'index': f'{save_dir}exp_index.p',
        'lengths': f'{save_dir}exp_doc_lengths.p',
        'docs': f'{save_dir}exp_documents.p'
    }

    documents, queries, relevance = read_data(data_path)
    if not engine.index_exists(paths=save_paths):
        print("* Building index... *")
        engine.build_index(docs=documents, paths=save_paths, dir=save_dir)
        print("* Index was built successfully! *")
    else:
        print("* Loading index... *")
        engine.load_index(paths=save_paths)
        print("* Index was loaded successfully! *")


if __name__ == '__main__':
    launch()