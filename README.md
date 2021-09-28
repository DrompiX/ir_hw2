# Document summarization & query expansion - Information Retrieval homework 2, Innopolis University

How to run the code?

* Download two folders (`data.nosync` and `engine_data`) from [my Google Drive](https://drive.google.com/drive/folders/1efq4vyEQHCWJBRAgS_Y0OU5HW8Go87of?usp=sharing)
* Put both folders in a root directory of the project
* Make sure that Python version is 3.7+
* Install all required packages by running `pip3 install -r requirements.txt` <br> <u>P.S</u>: better to use virtual environment

Now you can run the code by simply typing `python3 doc_sum.py` for document summarization task and `python3 query_exp.py` for query expansion one.
To provide any other query for document summarization, please consider changin code in `doc_sum.py` in line `query = "your query here"` in `launch()` function.

If you will have a problem with nltk (probably not loaded datasets), please use <br>

```
	import nltk
	nltk.download('wordnet')     # required for query expansion
	nltk.download('stopwords')   # required for both parts
```
