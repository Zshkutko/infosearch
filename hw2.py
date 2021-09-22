import os
from pymystem3 import Mystem
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


m = Mystem()
vectorizer = TfidfVectorizer()
curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, 'friends-data')
files_paths = []
episodes = []
clean_texts = []
corpus = []
filenames = []

for root, dirs, files in os.walk(data_dir):
    for name in files:
        filenames.append(name)
        files_paths.append(os.path.join(root, name))

for file_path in files_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    episodes.append(text)


def preprocessing(text):
    clean_episode = text.lower()
    for i in clean_episode:
        if i in string.punctuation:
            clean_episode = clean_episode.replace(i, '')
    clean_episode = clean_episode.replace('\n', ' ')
    clean_episode = clean_episode.replace('—', ' ')
    clean_episode = clean_episode.replace('  ', ' ')
    clean_episode = clean_episode.replace('\ufeff', '')
    clean_episode = re.sub(r'\d*', '', clean_episode)
    clean_episode = re.sub(r'[a-z][A-Z]*', '', clean_episode)
    clean_episode_list = []
    for j in clean_episode.split(' '):
        if j not in stopwords.words('russian'):
            clean_episode_list.append(j)
    lemmatized_episode = ''.join(m.lemmatize(' '.join(clean_episode_list)))
    return lemmatized_episode


for episode in episodes:
    corpus.append(preprocessing(episode))


def corpus_vectorizer(corpus): #возвращает матрицу Document-Term
    X = vectorizer.fit_transform(corpus)
    return X


def query_vectorizer(query): #возвращает вектор запроса
    preprocessed_query = preprocessing(query)
    query_vector = vectorizer.transform([preprocessed_query]).toarray()
    return query_vector


def cos_query(query, X): #подсчет близости
    cos_list = []
    query_vector = query_vectorizer(query)
    for x in X:
        cos = cosine_similarity(query_vector, x)[0][0]
        cos_list.append(cos)
    return cos_list


def main(query, X, filenames): #главная функция
    filenames_sorted = []
    ind = np.argsort(cos_query(query, X))[::-1]
    for i in ind:
        filenames_sorted.append(filenames[i])
    return filenames_sorted


if __name__ == '__main__':
    X = corpus_vectorizer(corpus)
    query = 'кошка'
    print(main(query, X, filenames))
