import json
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import numpy as np

count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')


def file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = list(f)[:100]
    best_answers = []
    for text in corpus:
        answers_dict = {}
        answers = json.loads(text)['answers']
        if answers:
            for answer in answers:
                if answer['text'] and answer['author_rating']['value']:
                    answers_dict[answer['text']] = int(answer['author_rating']['value'])
            best_answers.append(sorted(answers_dict.items(), key=lambda kv: kv[1])[-1][0])
    return best_answers


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
    text = ' '.join([morph.parse(x)[0].normal_form for x in clean_episode_list])
    return text


def make_indexing(texts):
    rows = []
    cols = []
    values = []

    x_count_vec = count_vectorizer.fit_transform(texts)
    x_tf_vec = tf_vectorizer.fit_transform(texts)
    x_tfidf_vec = tfidf_vectorizer.fit_transform(texts)

    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)
    tf = x_tf_vec

    k = 2
    b = 0.75

    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()

    B_1 = (k * (1 - b + b * len_d / avdl))
    B_1 = np.expand_dims(B_1, axis=-1)

    for i, j in zip(*tf.nonzero()):
        rows.append(i)
        cols.append(j)
        A = idf[0][j] * tf[i, j] * (k + 1)
        B = tf[i, j] + B_1[i]
        value = A / B
        values.append(value[0][0])

    return sparse.csr_matrix((values, (rows, cols)))


def get_query(query_str):
    query_prep = preprocessing(query_str)
    query_vec = tfidf_vectorizer.transform([query_prep])
    return query_vec


def search(corpus, query, matrix):
    corpus = np.array(corpus)
    scores = np.dot(matrix, query.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    return corpus[sorted_scores_indx.ravel()]


def main():
    corpus = file('data.jsonl')
    corpus_prep = []
    for text in corpus:
        corpus_prep.append(preprocessing(text))
    corpus_matrix = make_indexing(corpus_prep)
    print('Enter your question:')
    question = input()
    query = get_query(question)
    result = search(corpus, query, corpus_matrix)
    print('Top-5 answers:')
    for i in result[:5]:
        print('—', i)


if __name__ == '__main__':
    main()
