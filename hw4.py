import json
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import string
import re
import nltk
from tqdm import tqdm
import pickle
from nltk.corpus import stopwords
from scipy import sparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

tiny_bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
tiny_bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

fast_text_model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')


def file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = list(f)[:10]
    best_answers = []
    all_answers = []
    questions = []
    for text in corpus:
        answers_dict = {}
        answers = json.loads(text)['answers']
        questions.append(' '.join(json.loads(text)['question']) + ' '.join(json.loads(text)['comment']))
        if answers:
            for answer in answers:
                if answer['text'] and answer['author_rating']['value']:
                    all_answers.append(answer['text'])
                    answers_dict[answer['text']] = int(answer['author_rating']['value'])
            best_answers.append(sorted(answers_dict.items(), key=lambda kv: kv[1])[-1][0])
    return best_answers, all_answers, questions


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


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


def index_tf_idf(texts):
    return tfidf_vectorizer.fit_transform(texts)


def index_count_vec(texts):
    return count_vectorizer.fit_transform(texts)


def index_bm25(texts):
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


def index_tiny_bert(texts):
    tiny_embeddings = []
    for text in texts:
        tiny_embeddings.append(embed_bert_cls(text, tiny_bert_model, tiny_bert_tokenizer))
    return sparse.csr_matrix(tiny_embeddings)


def index_fast_text(texts):
    embs = []
    for text in texts:
        tokens = text.split()
        token_embeddings = np.zeros((len(tokens), fast_text_model.vector_size))
        for i, token in enumerate(tokens):
            token_embeddings[i] = fast_text_model[token]
        if token_embeddings.shape[0] != 0:
            mean_token_embs = np.mean(token_embeddings, axis=0)
            normalized_embeddings = mean_token_embs / np.linalg.norm(mean_token_embs)
            embs.append(normalized_embeddings)
    return sparse.csr_matrix(embs)


def query_bert(query):
    query_embs = []
    for q in query:
        query_emb = embed_bert_cls([q], tiny_bert_model, tiny_bert_tokenizer)
        query_embs.append(query_emb)
    return sparse.csr_matrix(query_embs)


def query_fast_text(query):
    return index_fast_text(query)


def query_tf_idf(query):
    return tfidf_vectorizer.transform(query)


def query_count_vec(query):
    return count_vectorizer.transform(query)


def query_bm25(query):
    return tfidf_vectorizer.transform(query)


def search(corpus, embeddings, query):
    scores = np.dot(embeddings, query.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus = np.array(corpus)[sorted_scores_indx.ravel()]
    return corpus


def save_my_index(filename, idx_func, qry_func, corpus, questions):
    answers_embeddings = idx_func(corpus)
    questions_embeddings = qry_func(questions)
    sparse.save_npz(f'index_data/{filename}_answers.npz', answers_embeddings)
    sparse.save_npz(f'index_data/{filename}_questions.npz', questions_embeddings)


def main():
    num_answers = 5
    choice = input('METHOD (BERT, FASTTEXT, TFIDF, COUNTVEC, BM25)')
    query = input('QUERY: ')
    corpus, _, questions = file('questions_about_love.jsonl')
    all_answers_prep = []
    questions_prep = []
    for text in tqdm(corpus):
        all_answers_prep.append(preprocessing(text))
    for text in tqdm(questions[:9671]):
        questions_prep.append(preprocessing(text))

    if choice == 'BERT':
        answers_embeddings = index_tiny_bert(corpus)
        query = query_bert(query)
        search_res = search(corpus, answers_embeddings, query)
        for ans in range(num_answers):
            print(search_res[ans])
    elif choice == 'FASTTEXT':
        query = preprocessing(query)
        answers_embeddings = index_fast_text(all_answers_prep)
        query = query_fast_text(query)
        search_res = search(corpus, answers_embeddings, query)
        for ans in range(num_answers):
            print(search_res[ans])
    elif choice == 'TFIDF':
        print('GO')
        query = preprocessing(query)
        answers_embeddings = index_tf_idf(all_answers_prep)
        print(all_answers_prep[:5])
        query = query_tf_idf([query])
        search_res = search(corpus, answers_embeddings, query)
        for ans in range(num_answers):
            print(search_res[ans])
    elif choice == 'COUNT_VEC':
        query = preprocessing(query)
        answers_embeddings = index_count_vec(all_answers_prep)
        query = query_count_vec([query])
        search_res = search(corpus, answers_embeddings, query)
        for ans in range(num_answers):
            print(search_res[ans])
    elif choice == 'BM25':
        query = preprocessing(query)
        answers_embeddings = index_bm25(all_answers_prep)
        query = query_bm25([query])
        search_res = search(corpus, answers_embeddings, query)
        for ans in range(num_answers):
            print(search_res[ans])
    elif choice == 'save':
        save_my_index('bert', index_tiny_bert, query_bert, corpus, questions)
        save_my_index('fasttext', index_fast_text, query_fast_text, all_answers_prep, questions_prep)
        save_my_index('count_vec', index_count_vec, query_count_vec, all_answers_prep, questions_prep)
        save_my_index('tf-idf', index_tf_idf, query_tf_idf, all_answers_prep, questions_prep)
        save_my_index('bm25', index_bm25, query_bm25, all_answers_prep, questions_prep)
        pickle.dump(tf_vectorizer, open('vec_data/bm25.pickle', 'wb'))
        pickle.dump(tf_vectorizer, open('vec_data/tf-idf.pickle', 'wb'))
        pickle.dump(tf_vectorizer, open('vec_data/count_vec.pickle', 'wb'))
    else:
        print('Выберите что нибудь!')
        input()


if __name__ == '__main__':
    main()
