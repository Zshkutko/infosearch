import numpy as np
from scipy import sparse
import os


def from_files():
    indexes = []
    questions = []
    for file in os.listdir('index_data'):
        if 'answers' in file:
            indexes.append(file)
        else:
            questions.append(file)
    return indexes, questions


def matrix_sort(index, query):
    scores = np.dot(index, query.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    return sorted_scores_indx


def compute_metric(index, query):
    sorted_scores_indx = matrix_sort(index, query)
    c = 0
    for index, row in enumerate(sorted_scores_indx):
        top_results = row[:5]
        if index in top_results:
            c += 1
    final_score = c / len(sorted_scores_indx)
    return final_score


def main():
    indexes_files, questions_files = from_files()
    final_scores = []
    for i in range(len(indexes_files)):
        index = sparse.load_npz('index_data/' + str(indexes_files[i]))
        query = sparse.load_npz('index_data/' + str(questions_files[i]))

        final_scores.append(compute_metric(index, query))

    for i in zip(indexes_files, final_scores):
        print(i)


if __name__ == '__main__':
    main()
