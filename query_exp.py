import json
import re
import numpy as np
import search_engine as engine
from search_engine import Article
from typing import List, Tuple, Dict
from collections import Counter

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
        if j - 1 < len(top_k_results):
            doc2relevance = combine_score_and_docs(top_k_results[j - 1], scores)
            dcg_score = DCG(doc2relevance, top_k)
            upd_score = list(map(lambda x: (x[0], 5 - x[1]), scores))
            idcg_score = DCG(sorted(upd_score, key=lambda _: _[1], reverse=True), top_k)
            ndcg_score += 0.0 if idcg_score == 0 else dcg_score / idcg_score
        
    ndcg_score /= len(relevance)

    return ndcg_score


def docs2vecs(docs: Dict[int, Article]):
    vectors = {}
    for doc_id, doc in docs.items():
        terms = engine.preprocess(str(doc))
        vectors[doc_id] = Counter()
        for term in terms:
            vectors[doc_id][term] += 1
            
    for doc_id in vectors:
        for term in vectors[doc_id]:
            if term in engine.index:
                idf = np.log10(len(engine.documents) / engine.index[term][0])
            else:
                idf = 0
            vectors[doc_id][term] *= idf
    
    return vectors


def rocchio(query: str, relevance: List[Tuple[int, int]],
            top_docs: Dict[int, Article], alph=1.0, beta=0.75, gamma=0.15):
    top_docs_vectors = docs2vecs(top_docs)
    query_vector = Counter(engine.preprocess(query))
    new_query = dict((k, v * alph) for k, v in query_vector.items())
    
    center = dict((k, 0) for k in query_vector.keys())
    relevant_docs = set()
    for doc_id, _ in relevance:
        if doc_id in top_docs_vectors:
            relevant_docs.add(doc_id)
            for term in top_docs_vectors[doc_id]:
                # if top_docs_vectors[doc_id][term] > 0:
                if term in center:
                    center[term] += top_docs_vectors[doc_id][term]
                else:
                    center[term] = top_docs_vectors[doc_id][term]

    neg_center = dict((k, 0) for k in query_vector.keys())
    for doc_id in top_docs_vectors:
        if doc_id not in relevant_docs:
            for term in top_docs_vectors[doc_id]:
                if term in neg_center:
                    neg_center[term] += top_docs_vectors[doc_id][term]
                else:
                    neg_center[term] = top_docs_vectors[doc_id][term]
    
    if len(relevant_docs) == 0:
        return new_query

    for term in center:
        if term in new_query:
            new_query[term] += beta * 1 / len(relevant_docs) * center[term]
        else:
            new_query[term] = beta * 1 / len(relevant_docs) * center[term]
    
    non_rel_cnt = len(top_docs_vectors) - len(relevant_docs)
    if gamma > 0 and non_rel_cnt > 0:
        for term in neg_center:
            if term in new_query:
                new_query[term] -= gamma * 1 / non_rel_cnt * neg_center[term]
                new_query[term] = max(0, new_query[term])
    
    return new_query


def pseudo_relevance_feedback(query: str, top_docs: Dict[int, Article],
                              relevant_n=5, alph=1.0, beta=0.75, gamma=0):
    relevance = []
    relevant_cnt = min(int(len(top_docs) / 2), relevant_n)
    for doc_id in top_docs:
        if relevant_cnt == 0:
            break
        else:
            relevance.append((doc_id, 1))
            relevant_cnt -= 1
    
    return rocchio(query, relevance, top_docs, alph, beta, gamma)


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
    
    top_k_results = []
    top_k_rf = []
    top_k_prf = []
    for q_id in queries:
        if q_id != 0:
            q_results = engine.answer_query(queries[q_id], 15, get_ids=True)
            top_k_results.append(list(q_results.keys()))

            rf_query = rocchio(queries[q_id], relevance[q_id], q_results, beta=0.75, gamma=0.15)
            rf_q_results = engine.answer_query(rf_query, 15, get_ids=True, is_raw=False)
            top_k_rf.append(list(rf_q_results.keys()))

            prf_query = pseudo_relevance_feedback(queries[q_id], q_results, 5, beta=0.75, gamma=0.0)
            prf_q_results = engine.answer_query(prf_query, 15, get_ids=True, is_raw=False)
            top_k_prf.append(list(prf_q_results.keys()))

    print('Raw query NDCG:', NDCG(top_k_results, relevance, 15))
    print('Relevance feedback (Rocchio) NDCG:', NDCG(top_k_rf, relevance, 15))
    print('Pseudo relevance feedback (Rocchio) NDCG:', NDCG(top_k_prf, relevance, 15))


if __name__ == '__main__':
    launch()