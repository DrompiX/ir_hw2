import pandas as pd

articles = pd.read_csv("data.nosync/articles1.csv", nrows=50000)

articles_cut = articles[['title', 'content']]

articles_cut.to_csv("data.nosync/articles50000.csv", index_label='id')
