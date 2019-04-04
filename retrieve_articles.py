import pandas as pd

articles = pd.read_csv("data.nosync/articles1.csv", nrows=1000)

articles_cut = articles[['title', 'content']]

articles_cut.to_csv("data.nosync/articles1000.csv", index_label='id')
