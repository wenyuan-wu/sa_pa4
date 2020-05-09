#! /usr/bin/python
# -*- coding: utf-8 -*-
# Author: Wenyuan Wu, 18746867
# Date: 08.05.2020
# Additional Info:
# To get help, type in command line:
# $ python pa4.py -h
# Exemplary usage:
# $ python pa4.py -i pa4_Q.txt

import numpy as np
import gensim.downloader as api
import csv
import json
import argparse


word_vectors = api.load('word2vec-google-news-300')


def get_average_vector(sent: str) -> np.ndarray:
    """To calculate the average vector of a sentence, simply to take average of all word vectors elementwise."""
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


def get_avg_vec_dict(sent_list: list) -> dict:
    """Return a dictionary of average vectors of all sentences list."""
    avg_vec_dict = {}
    for key in sent_list:
        avg_vec_dict[key] = get_average_vector(key)
    return avg_vec_dict


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """To calculate cosine distance between two vectors."""
    return 1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_baseline_distance(q_list: list, c_list: list) -> dict:
    """To calculate distance of entities from two lists by baseline approach."""
    result_dict = {}
    q_avg_vec = get_avg_vec_dict(q_list)
    c_avg_vec = get_avg_vec_dict(c_list)
    for query in q_list:
        temp_dict = {}
        for comp in c_list:
            temp_dict[comp] = cosine_distance(c_avg_vec[comp], q_avg_vec[query])
        temp_dict = {k: v for k, v in sorted(temp_dict.items(), key=lambda item: item[1])}
        result_dict[query] = temp_dict
    return result_dict


def get_wmd_distance(q_list: list, c_list: list) -> dict:
    """To calculate distance of entities from two lists by WMD approach."""
    result_dict = {}
    for query in q_list:
        temp_dict = {}
        for comp in c_list:
            temp_dict[comp] = word_vectors.wmdistance(comp.lower().split(), query.lower().split())
        temp_dict = {k: v for k, v in sorted(temp_dict.items(), key=lambda item: item[1])}
        result_dict[query] = temp_dict
    return result_dict


def get_rank_from_dict(q_list: list, c_list: list, dist_dict: dict) -> dict:
    """To convert distance dictionary into ranking dictionary."""
    result_dict = {}
    for query in q_list:
        rank_list = list(dist_dict[query].keys())
        temp_dict = {}
        for comp in c_list:
            temp_dict[comp] = rank_list.index(comp) + 1
        result_dict[query] = temp_dict
    return result_dict


def print_ranking_table(q_list: list, q_rank_dict: dict, c_list: list, c_rank_dict: dict, baseline: dict, wmd: dict):
    """To print out the ranking results of two approaches in form of tables."""
    for query in q_list:
        print('query: {}'.format(query))
        print('correct rank: {}\n'.format(str(q_rank_dict[query])))
        print('{:33}{:10}{}'.format('ranking', 'baseline', 'WMD'))
        for comp in c_list:
            print('{:3}{:30}{:10}{}'.format(str(c_rank_dict[comp]),
                                            comp,
                                            str(baseline[query][comp]),
                                            str(wmd[query][comp])))
        print('\n')


def get_argument_parser() -> argparse.ArgumentParser:
    """To create an argument parser with required options."""
    parser = argparse.ArgumentParser(
        description='A Python script to compare the results of '
                    'classification with sparse vs. dense vectors')
    parser.add_argument('-i', help='input file', required=True)
    return parser


def main():
    # To get the list of competences
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

    # To get list of queries
    args = get_argument_parser().parse_args()
    query_rank_dict = {}
    with open(args.i, 'r', newline='') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for row in reader:
            query_rank_dict[row[0]] = int(row[1])
    query_list = list(query_rank_dict.keys())

    # To calculate results by two different approaches and save to json file
    distance_baseline = get_baseline_distance(query_list, comp_list)
    with open('distance_baseline.json', 'w') as fp:
        json.dump(distance_baseline, fp, indent=4)

    distance_wmd = get_wmd_distance(query_list, comp_list)
    with open('distance_wmd.json', 'w') as fp:
        json.dump(distance_wmd, fp, indent=4)

    ranking_baseline = get_rank_from_dict(query_list, comp_list, distance_baseline)
    with open('ranking_baseline.json', 'w') as fp:
        json.dump(ranking_baseline, fp, indent=4)

    ranking_wmd = get_rank_from_dict(query_list, comp_list, distance_wmd)
    with open('ranking_wmd.json', 'w') as fp:
        json.dump(ranking_wmd, fp, indent=4)

    # print final result as standard output
    print_ranking_table(query_list, query_rank_dict, comp_list, comp_rank_dict, ranking_baseline, ranking_wmd)


if __name__ == '__main__':
    main()
