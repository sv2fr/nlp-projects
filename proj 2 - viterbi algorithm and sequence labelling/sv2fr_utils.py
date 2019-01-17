# -*- coding: utf-8 -*-
"""
Created on 28 Oct 2018

@author: Sri
"""

import pandas as pd
import csv


def read_file_into_df(filename):
    """
    Read file into pandas dataframe
    :param filename: name of the file
    :return: dataframe
    """
    return pd.read_table(filename, header=None)


def split_by_space(x):
    """
    Split the dataframe rows by space
    :param x: dataframe of rows
    :return: dataframe of rows in which each row is split by space
    """
    return [row.split(' ') for row in x]


def split_by_slash(x):
    """
    Split the dataframe rows by /
    :param x: dataframe of rows
    :return: dataframe of rows in which each row is split by /
    """
    words_master, tokens_master = [], []
    for row in x:
        words, tokens = [], []
        for word in row:
            items = word.split('/')
            words.append(items[0])
            tokens.append(items[1])
        words_master.append(words)
        tokens_master.append(tokens)
    return words_master, tokens_master


def get_all_words(x):
    """
    Get all the words from list in each row of dataframe
    :param x: dataframe of rows
    :return: master list of all words
    """
    return [word for row in x for word in row]


def replace_least_common_with_unk(x, freq_count, cutoff):
    """
    Replace least frequent words in dataframe (defined by the cutoff) with Unk
    :param x: dataframe of rows
    :param freq_count: the dictionary containing word: freq pairs
    :param cutoff: the cutoff value for frequency
    :return: dataframe with less frequent words replaced with Unk
    """
    words_master = []
    for row in x:
        words = []
        for word in row:
            if freq_count[word] <= cutoff:
                words.append('Unk')
            else:
                words.append(word)
        words_master.append(words)
    return words_master


def replace_non_vocab_words_with_unk(x, vocab):
    """
    Replace words not in vocab with Unk
    :param x: dataframe of rows
    :param vocab: set of words in vocab
    :return: dataframe of rows with words not in vocab replaced with Unk
    """
    words_master = []
    for row in x:
        words = []
        for word in row:
            if word not in vocab:
                words.append('Unk')
            else:
                words.append(word)
        words_master.append(words)
    return words_master


def get_all_tags(x):
    """
    Get the list of all tags from dataframe
    :param x: dataframe of rows
    :return: list of unique tags
    """
    tags = list(set([word for row in x for word in row]))
    tags.sort()
    return tags


def get_zero_dataframe(index, columns):
    """
    Get empty dataframe initialized with zeros
    :param index: index for dataframe
    :param columns: columns for dataframe
    :return: dataframe defined by index and columns filled with 0
    """
    return pd.DataFrame(0.0, index=index, columns=columns)


def add_start_end_tags(x):
    """
    Add Start and End tags to each row in dataframe
    :param x: dataframe
    :return: dataframe with each row appended with Start and End
    """
    for row in x:
        row.insert(0, 'Start')
        row.append('End')


def remove_start_end_tags(x):
    """
    Remove Start and End tags from each row of dataframe
    :param x: dataframe
    :return: dataframe with Start and End tags removed from each row
    """
    for row in x:
        row.remove('Start')
        row.remove('End')


def write_to_csv(x, filename):
    """
    Writes dataframe into csv file with each row containing (row, column, value)
    :param x: the dataframe
    :param filename: name of the file to write to
    :return:
    """
    rows = []
    [rows.append([row, column, x.loc[row][column]]) for row in x.index for column in x.columns]

    with open(filename, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(rows)


def count_matches(list_a, list_b):
    """
    Return the count of element-wise match in two lists
    :param list_a: list to be compared
    :param list_b: list to be compared
    :return:
    """

    return sum(a == b for a, b in zip(list_a, list_b))


def combine_by_slash(list_a, list_b):
    """
    Combine each element of list_a with list_b in the format element_list_a/element_list_b
    :param list_a: list to be combined
    :param list_b: list to be combined
    :return:
    """

    slash = '/'
    return ' '.join([str(a) + slash + str(b) for a, b in zip(list_a, list_b)])
