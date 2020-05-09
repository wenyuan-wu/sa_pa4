#! /usr/bin/python
# -*- coding: utf-8 -*-
# Author: Wenyuan Wu, 18746867
# Date: 08.05.2020
# Additional Info:
# Install packages:
# pip install -r requirements.txt
# To get help, type in command line:
# $ python pa4.py -h
# Exemplary usage:
# $ python pa4.py -i pa4_Q.txt

import numpy as np
import gensim.downloader as api
import csv
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


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """To calculate cosine distance between two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_baseline_dict(q_list: list, c_list: list) -> dict:
    """To calculate distance of entities from two lists by baseline approach."""
    result_dict = {}
    q_avg_vec = get_avg_vec_dict(q_list)
    c_avg_vec = get_avg_vec_dict(c_list)
    for query in q_list:
        temp_dict = {}
        for comp in c_list:
            temp_dict[comp] = cosine_similarity(c_avg_vec[comp], q_avg_vec[query])
        temp_dict = {k: v for k, v in sorted(temp_dict.items(), key=lambda item: item[1], reverse=True)}
        result_dict[query] = temp_dict
    return result_dict


def get_wmd_dict(q_list: list, c_list: list) -> dict:
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
    """Convert distance dictionary into ranking dictionary."""
    result_dict = {}
    for query in q_list:
        rank_list = list(dist_dict[query].keys())
        temp_dict = {}
        for comp in c_list:
            temp_dict[comp] = rank_list.index(comp) + 1
        result_dict[query] = temp_dict
    return result_dict


def print_ranking_table(q_list: list, q_rank_dict: dict, c_list: list, c_rank_dict: dict, baseline: dict, wmd: dict):
    """Print out the ranking results of two approaches in form of tables."""
    for idx, query in enumerate(q_list):
        print('Query {}: {}'.format(idx + 1, query))
        true_rank = q_rank_dict[query]
        print('Correct rank: {}\n'.format(str(true_rank)))
        print('{:34}{:10}{}'.format(' Ranking', ' Baseline', ' WMD'))
        for comp in c_list:
            baseline_rank = baseline[query][comp]
            if baseline_rank in [1, 2, 3]:
                baseline_rank = '>' + str(baseline_rank)
            else:
                baseline_rank = ' ' + str(baseline_rank)
            wmd_rank = wmd[query][comp]
            if wmd_rank in [1, 2, 3]:
                wmd_rank = '>' + str(wmd_rank)
            else:
                wmd_rank = ' ' + str(wmd_rank)
            comp_rank = c_rank_dict[comp]
            if true_rank == comp_rank:
                comp_rank = '>' + str(comp_rank)
            else:
                comp_rank = ' ' + str(comp_rank)

            print('{:4}{:30}{:10}{}'.format(comp_rank, comp, baseline_rank, wmd_rank))
        print('\n')


def get_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser with required options."""
    parser = argparse.ArgumentParser(
        description='A Python script to compare the results of '
                    'classification with sparse vs. dense vectors')
    parser.add_argument('-i', help='input file', required=True)
    return parser


def dict_to_file(filepath: str, input_dict: dict) -> None:
    """Save dictionary into file for inspection."""
    with open(filepath, 'w') as fp:
        for key, val in input_dict.items():
            fp.write(key)
            fp.write('\n')
            for k, v in val.items():
                fp.write('{}: {}'.format(k, v))
                fp.write('\n')
            fp.write('\n')


def main():
    # get the list of competences
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

    # get list of queries
    args = get_argument_parser().parse_args()
    query_rank_dict = {}
    with open(args.i, 'r', newline='') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for row in reader:
            query_rank_dict[row[0]] = int(row[1])
    query_list = list(query_rank_dict.keys())

    # calculate results by two different approaches and save to json format file for inspection purpose
    dict_baseline = get_baseline_dict(query_list, comp_list)
    dict_to_file('dict_baseline.txt', dict_baseline)

    dict_wmd = get_wmd_dict(query_list, comp_list)
    dict_to_file('dict_wmd.txt', dict_baseline)

    ranking_baseline = get_rank_from_dict(query_list, comp_list, dict_baseline)
    dict_to_file('ranking_baseline.txt', ranking_baseline)

    ranking_wmd = get_rank_from_dict(query_list, comp_list, dict_wmd)
    dict_to_file('ranking_wmd.txt', ranking_wmd)

    # print final result as standard output
    print_ranking_table(query_list, query_rank_dict, comp_list, comp_rank_dict, ranking_baseline, ranking_wmd)


if __name__ == '__main__':
    main()
