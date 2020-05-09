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
import json
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
    return parser


word_vectors = api.load('word2vec-google-news-300')

comp_rank_dict = {
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
comp_list = list(comp_rank_dict.keys())


def get_average_vector(sent: str) -> np.ndarray:
    vector = np.zeros(300)
    for idx, word in enumerate(sent.lower().split()):
        try:
            new_vector = word_vectors[word]
        except KeyError:
            new_vector = np.zeros(300)
        if idx == 0:
            vector = new_vector
        else:
            avg_vector = np.add(vector, new_vector) / 2.0
            vector = avg_vector
    return vector


comp_avg_vec = {}
for key in comp_rank_dict.keys():
    comp_avg_vec[key] = get_average_vector(key)

query_rank_dict = {}
with open('pa4_Q.txt', 'r', newline='') as infile:
    reader = csv.reader(infile, delimiter='\t')
    for row in reader:
        query_rank_dict[row[0]] = int(row[1])
query_list = list(query_rank_dict.keys())

query_avg_vec = {}
for key in query_rank_dict.keys():
    query_avg_vec[key] = get_average_vector(key)


def get_baseline_distance(query_list: list) -> dict:
    result_dict = {}
    for query in query_list:
        temp_dict = {}
        for comp in comp_list:
            temp_dict[comp] = cosine_distance(comp_avg_vec[comp], query_avg_vec[query])
        temp_dict = {k: v for k, v in sorted(temp_dict.items(), key=lambda item: item[1])}
        result_dict[query] = temp_dict
    return result_dict


def get_wmd_distance(query_list: list) -> dict:
    result_dict = {}
    for query in query_list:
        temp_dict = {}
        for comp in comp_list:
            temp_dict[comp] = word_vectors.wmdistance(comp.lower().split(), query.lower().split())
        temp_dict = {k: v for k, v in sorted(temp_dict.items(), key=lambda item: item[1])}
        result_dict[query] = temp_dict
    return result_dict


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate cosine similarity of two vectors.

    Parameters
    ----------
    # TODO
    x: Pandas Series
    y: Pandas Series

    Returns
    -------
    A float number
    """
    return 1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


distance_baseline = get_baseline_distance(list(query_avg_vec.keys()))

with open('distance_baseline.json', 'w') as fp:
    json.dump(distance_baseline, fp, indent=4)


distance_wmd = get_wmd_distance(list(query_avg_vec.keys()))

with open('distance_wmd.json', 'w') as fp:
    json.dump(distance_wmd, fp, indent=4)


def get_rank_from_dict(dist_dict: dict) -> dict:
    result_dict = {}
    for query in query_rank_dict.keys():
        rank_list = list(dist_dict[query].keys())
        temp_dict = {}
        for comp in comp_list:
            temp_dict[comp] = rank_list.index(comp) + 1
        result_dict[query] = temp_dict
    return result_dict


ranking_baseline = get_rank_from_dict(distance_baseline)
with open('ranking_baseline.json', 'w') as fp:
    json.dump(ranking_baseline, fp, indent=4)


ranking_wmd = get_rank_from_dict(distance_wmd)
with open('ranking_wmd.json', 'w') as fp:
    json.dump(ranking_wmd, fp, indent=4)


def print_ranking_table(baseline, wmd):
    for query in query_list:
        print('query: {}'.format(query))
        print('correct rank: {}\n'.format(str(query_rank_dict[query])))
        print('{:33}{:10}{}'.format('ranking', 'baseline', 'WMD'))
        for comp in comp_list:
            print('{:3}{:30}{:10}{}'.format(str(comp_rank_dict[comp]),
                                            comp,
                                            str(baseline[query][comp]),
                                            str(wmd[query][comp])))
        print('\n')


print_ranking_table(ranking_baseline, ranking_wmd)


# def main():
#     pass
#
# 
# if __name__ == '__main__':
#     main()
