import csv
from collections import namedtuple

Article = namedtuple('Article', ['title', 'text'])

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
            result.append(Article(title=article[1], text=article[2]))
    
    return result

def launch():
    print("Doc sum launcher")
    data_path = 'data.nosync/articles1000.csv'
    articles = read_data(data_path)
    print(articles[0])

if __name__ == '__main__':
    launch()