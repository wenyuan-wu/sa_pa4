#! /usr/bin/python
# -*- coding: utf-8 -*-
# Author: Wenyuan Wu, 18746867
# Date: 19.04.2020
# Additional Info:
# ### PA3: Classification with sparse vs. dense vectors
# To get help, type in command line:
# $ python pa3.py -h
# Exemplary usage:
# $ python pa3.py -i pa3_input_text.txt -b pa3_B.txt -t pa3_T.txt


import numpy as np
import gensim.downloader as api
import csv
import argparse


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser with required options.

    Returns
    -------
    The argument parser with all arguments added.
    """
    parser = argparse.ArgumentParser(
        description='A Python script to compare the results of '
                    'classification with sparse vs. dense vectors')
    parser.add_argument('-i', help='input file', required=True)
    parser.add_argument('-b', help='base file', required=True)
    parser.add_argument('-t', help='target file', required=True)
    return parser


word_vectors = api.load('word2vec-google-news-300')

competence_rank_dict = {
    'Clean kitchen': 1,
    'Teach basic cooking': 2,
    'Chinese gourmet cuisine': 3,
    'Clean outdoor seating': 4,
    'Cultivate garden': 5,
    'Carry out transports': 6,
    'Maintain laundry': 7,
    'Sewing': 8,
    'Do shopping': 9,
    'Baby care': 10,
    'Look after toddlers': 11,
    'Advise child care': 12,
    'Dog training': 13,
    'Walk a dog': 14,
    'Do painting work': 15,
    'Provide knowledge of wine': 16,
}


def get_average_vector(sent: str) -> np.ndarray:
    vector = np.zeros(300)
    for idx, word in enumerate(sent.lower().split()):
        new_vector = word_vectors[word]
        if idx == 0:
            vector = new_vector
        else:
            avg_vector = np.add(vector, new_vector) / 2.0
            vector = avg_vector
    return vector


competence_avg_vec = {}
for key in competence_rank_dict.keys():
    competence_avg_vec[key] = get_average_vector(key)

query_rank_dict = {}
with open('pa4_Q.txt', 'r', newline='') as infile:
    reader = csv.reader(infile, delimiter='\t')
    for row in reader:
        query_rank_dict[row[0]] = int(row[1])


def main():
    pass


if __name__ == '__main__':
    main()
