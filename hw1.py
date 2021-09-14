import os
from pymystem3 import Mystem
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

vectorizer = CountVectorizer(analyzer='sword')
m = Mystem()
curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, 'friends-data')
files_paths = []
episodes = []
clean_texts = []
corpus = []

for root, dirs, files in os.walk(data_dir):
    for name in files:
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

X = vectorizer.fit_transform(corpus)

lemmas_list = vectorizer.get_feature_names()
matrix_freq = np.asarray(X.sum(axis=0)).ravel()
freq_list = matrix_freq.tolist()
freq_dict = dict(zip(lemmas_list, freq_list))
print('Самое частотное слово: ', max(freq_dict, key=freq_dict.get))
print('Самое редкое слово: ', min(freq_dict, key=freq_dict.get))

matrix = X.toarray()
columns = lemmas_list
df = pd.DataFrame(matrix, columns=columns)
df_new = df.loc[:, ~(df == 0).any()]
words_indexes = df_new.columns.tolist()
print('Слова, встречающиеся во всех документах: ', ', '.join(words_indexes))

names = ['моника', 'мон', 'рэйчел', 'рейч', 'чендлер', 'чэндлер', 'чен',
         'фиби', 'фибс', 'росс', 'джоуи', 'джо']
chars_freq = {}
for name in names:
    chars_freq[name] = int(freq_dict.get(name))
print('Самый популярный герой: ', max(chars_freq, key=chars_freq.get))
