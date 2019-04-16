import json
import re
import sys
import random
import numpy as np
import search_engine as engine
from search_engine import Article
from typing import List, Tuple, Dict
from collections import Counter
from nltk.corpus import wordnet, stopwords
# import nltk
# nltk.download('wordnet')

stop_words = stopwords.words('english')


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


def preprocess(text: str, remove_stop: bool=True) -> str:
    '''Preprocess text and possibly remove stopwords
    
    Args:
        text: text to be preprocessed
        remove_stop: whether to remove stop words
    
    Returns:
        preprocessed text
    '''
    return [t for t in engine.preprocess(text) if t not in stop_words]


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

    :param top_k_results: list of lists of ranked results for each query [[doc_id1, doc_id2,...], ...]
                          the i-th result corresponds to (i+1)-th query_id. There may be less than top_k
                          results returned for a query, but never more.
    :param relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    :param top_k: (max) number of results retrieved for each query, use this value to find normalization
                  factor for each query
    :return: NDCG score
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
    '''Converts documents to vector representation
    
    Args:
        docs: documents to be converted
    Returns:
        resulting vectors
    '''
    vectors = {}
    for doc_id, doc in docs.items():
        terms = preprocess(str(doc))
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
    '''Implementation of Rocchio algorithm

    Args:
        query: input query
        relevance: list of relevant docs for query
        top_docs: top k docs for query
        alph: weight of original query
        beta: weight of relevant docs
        gamma: weight of irrelevant docs
    Return:
        modified query as Dict (term: score)
    '''
    top_docs_vectors = docs2vecs(top_docs)
    query_vector = Counter(preprocess(query))
    new_query = dict((k, v * alph) for k, v in query_vector.items())
    
    # center for relevant docs
    center = dict((k, 0) for k in query_vector.keys()) 
    relevant_docs = set()
    for doc_id, _ in relevance:
        if doc_id in top_docs_vectors:
            relevant_docs.add(doc_id)
            for term in top_docs_vectors[doc_id]:
                if term in center:
                    center[term] += top_docs_vectors[doc_id][term]
                else:
                    center[term] = top_docs_vectors[doc_id][term]

    # center for irrelevant docs
    neg_center = dict((k, 0) for k in query_vector.keys())
    for doc_id in top_docs_vectors:
        if doc_id not in relevant_docs:
            for term in top_docs_vectors[doc_id]:
                if term in neg_center:
                    neg_center[term] += top_docs_vectors[doc_id][term]
                else:
                    neg_center[term] = top_docs_vectors[doc_id][term]
    
    # if no relevant docs, return same query
    if len(relevant_docs) == 0:
        return new_query

    term_candidates = {}
    # recalculate weights for terms and add new
    for term in center:
        if term in term_candidates:
        # if term in new_query:
            # new_query[term] += beta * 1 / len(relevant_docs) * center[term]
            term_candidates[term] += beta * 1 / len(relevant_docs) * center[term]
        else:
            # new_query[term] = beta * 1 / len(relevant_docs) * center[term]
            term_candidates[term] = beta * 1 / len(relevant_docs) * center[term]
    
    term_candidates = sorted(term_candidates.items(), key=lambda item: item[1], reverse=True)
    for term_score in term_candidates[:3]:
        new_query[term_score[0]] = term_score[1]

    non_rel_cnt = len(top_docs_vectors) - len(relevant_docs)
    if gamma > 0 and non_rel_cnt > 0:
        for term in neg_center:
            if term in new_query:
                new_query[term] -= gamma * 1 / non_rel_cnt * neg_center[term]
                new_query[term] = max(0, new_query[term])
    
    return new_query


def get_k_relevant_docs(docs: Dict[int, Article], k: int):
    ''' Returns relevance for top k relevant docs

    Args:
        docs: considered docs
        k: amount of docs to return
    Returns:
        list of k tuples: (doc_id, 1)
    '''
    relevance = []
    relevant_cnt = min(int(len(docs) / 2), k)
    for doc_id in docs:
        if relevant_cnt == 0:
            break
        else:
            relevance.append((doc_id, 1))
            relevant_cnt -= 1
    
    return relevance


def pseudo_relevance_feedback(query: str, top_docs: Dict[int, Article],
                              relevant_n=5, alph=1.0, beta=0.75, gamma=0):
    '''Implementation of pseudo relevance feedback
    
    Based on implementation of roccio algorithm

    Args:
        query: input query
        top_docs: top k docs for query
        relevant_n: number of first docs to consider relevant
        alph: weight of original query
        beta: weight of relevant docs
        gamma: weight of irrelevant docs
    Return:
        modified query as Dict (term: score)
    '''
    relevance = get_k_relevant_docs(top_docs, relevant_n)
    return rocchio(query, relevance, top_docs, alph, beta, gamma)


def calculate_relation(term1_syns, term2_syns):
    '''Calculates relation between definitions of two terms'''
    if len(term1_syns) > 0 and len(term2_syns) > 0:
        return wordnet.wup_similarity(term1_syns[0], term2_syns[0])
    else:
        return 0


def global_wordnet_exp(query: str, relevant_docs, add_terms: int=3):
    '''Implementation of global query expansion method using wordnet

    Considers selection of top `add_terms` terms from pseudo
    relevant documents via matching combination of 
    definitions from wordnet and other metrics (such as idf)
    P.S: CET - Candidate Expansion Term

    Args:
        query: initial query
        relevant_docs: pseudo relevant docs for query
        add_terms: number of terms to add to query
    Returns:
        new query
    '''
    all_terms = set()
    term2doc = {}
    # find all terms and save occurrences of terms in docs
    for doc_id, doc in relevant_docs.items():
        terms = set(preprocess(str(doc)))
        all_terms = all_terms | terms
        for term in terms:
            if term in term2doc:
                term2doc[term].append(doc_id)
            else:
                term2doc[term] = [doc_id]
    
    # calculate idf for each CET
    idf = {}
    for term in all_terms:
        N = len(engine.documents)
        N_t = len(engine.index[term]) - 1 if term in engine.index else 0
        idf[term] = max(1e-4, np.log10((N - N_t + 0.5) / (N_t + 0.5)))
    
    relations = {}
    terms = preprocess(query)
   
    # calculate relation score between each CET and query term 
    for cet_term in all_terms:
        for term in terms:
            _cet = wordnet.synsets(cet_term)
            _term = wordnet.synsets(cet_term)
            if cet_term not in relations:
                relations[cet_term] = {}
            relations[cet_term][term] = calculate_relation(_cet, _term)
    
    # Get scores for each doc for query
    doc_scores = engine.okapi_scoring(Counter(engine.preprocess(query)),
                                      engine.doc_lengths, 
                                      engine.index)
    doc_scores = dict((k, v) for k, v in doc_scores.items() if k in relevant_docs)
    max_score = max(doc_scores.values())
    doc_scores = dict((k, v / max_score) for k, v in doc_scores.items())
    
    # compute score for each CET
    cet2score = {}
    for cet_term in all_terms:
        for term in terms:
            similarity = [v for k, v in doc_scores.items() if term in term2doc and k in term2doc[term]]
            rel = relations[cet_term][term]
            idf_ = idf[cet_term]
            if cet_term in cet2score:
                cet2score[cet_term] += rel * idf_ * sum(similarity)
            else:
                cet2score[cet_term] = rel * idf_ * sum(similarity)
            cet2score[cet_term] /= (1 + cet2score[cet_term])
    
    # sort CET scores
    cet_scores = sorted(cet2score.items(), key=lambda item: item[1], reverse=True)
    
    # select CET's with biggest scores to add into new query
    new_query = [query]
    cnt = 0
    for word, _ in cet_scores:
        if cnt == add_terms:
            break
        # not include terms already in query
        if word not in terms:
            new_query.append(' ' + word)
            cnt += 1
    
    return ''.join(new_query)


def k_relevant(docs: Dict[int, Article], k: int):
    '''Returns first k documents
    
    Args:
        docs: input documents
        k: number of documents to return
    '''
    relevant = {}
    i = 0
    for doc_id, doc in docs.items():
        if i < k:
            relevant[doc_id] = doc
            i += 1
    
    return relevant


def train_test_split(docs: Dict[int, Article]):
    '''Splits input docs on train and test parts 50:50
    
    Args:
        docs: docs to be splitted
    Returns:
        ids for train and test parts
    '''
    document_ids = list(docs.keys())
    random.shuffle(document_ids)

    half = int(len(document_ids) / 2)
    train_docs = document_ids[:half]
    test_docs = document_ids[half:]

    return train_docs, test_docs


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

    train_ids, test_ids = train_test_split(documents)

    top_k_results = []
    top_k_rf = []
    top_k_prf = []
    top_k_glob = []
    processed = 1
    top_k = 10
    for q_id in queries:
        if q_id != 0:
            sys.stdout.write(f"\rQueries processed - {processed}/{len(queries) - 1}")
            sys.stdout.flush()

            q_results = engine.answer_query(queries[q_id], top_k, get_ids=True,
                                            included_docs=train_ids)
            
            q_results_test = engine.answer_query(queries[q_id], top_k, get_ids=True,
                                                 included_docs=test_ids)
            top_k_results.append(list(q_results_test.keys()))

            rf_query = rocchio(queries[q_id], relevance[q_id], q_results, beta=0.75, gamma=0.15)
            rf_q_results = engine.answer_query(rf_query, top_k, get_ids=True, is_raw=False,
                                               included_docs=test_ids)
            top_k_rf.append(list(rf_q_results.keys()))

            prf_query = pseudo_relevance_feedback(queries[q_id], q_results, 5, beta=0.75, gamma=0.15)
            prf_q_results = engine.answer_query(prf_query, top_k, get_ids=True, is_raw=False,
                                                included_docs=test_ids)
            top_k_prf.append(list(prf_q_results.keys()))

            glob_query = global_wordnet_exp(queries[q_id], k_relevant(q_results, 5), add_terms=2)
            glob_results = engine.answer_query(glob_query, top_k, get_ids=True,
                                               included_docs=test_ids)
            top_k_glob.append(list(glob_results.keys()))

            processed += 1

    print('\nRaw query NDCG:', NDCG(top_k_results, relevance, top_k))
    print('Relevance feedback (Rocchio) NDCG:', NDCG(top_k_rf, relevance, top_k))
    print('Pseudo relevance feedback (Rocchio) NDCG:', NDCG(top_k_prf, relevance, top_k))
    print('Global NDCG:', NDCG(top_k_glob, relevance, top_k))


if __name__ == '__main__':
    launch()
