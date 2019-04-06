import csv
from collections import namedtuple
import search_engine as engine

Article = namedtuple('Article', ['title', 'body'])

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

def doc_sum_1(doc, query):
    pass

def doc_sum_2(doc, query):
    pass

def doc_sum_3(doc, query):
    pass

def compare_doc_sum(query, doc, summary_len=50):
    sum_methods = [doc_sum_1, doc_sum_2, doc_sum_3]
    for method in sum_methods:
        print(f"Document summary for {method.__name__}")
        # method([], query)
    pass

def launch():
    print("Doc sum launcher")
    data_path = 'data.nosync/articles1000.csv'
    save_paths = {
        'index': 'index.p',
        'lengths': 'doc_lengths.p',
        'docs': 'documents.p'
    }
    
    if not engine.index_exists(paths=save_paths):
        print("* Building index... *")
        articles = read_data(data_path)
        engine.build_index(docs=articles, paths=save_paths)
        print("* Index was built successfully! *")
    else:
        print("* Loading index... *")
        engine.load_index(paths=save_paths)
        print("* Index was loaded successfully! *")
    
    q = "Tesla model X"
    docs = engine.answer_query(q, 2)
    compare_doc_sum(q, docs[0], 50)

if __name__ == '__main__':
    launch()