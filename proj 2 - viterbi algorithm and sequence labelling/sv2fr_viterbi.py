# -*- coding: utf-8 -*-
"""
Created on 28 Oct 2018

@author: Sri
"""

import pandas as pd
import numpy as np


class Viterbi(object):
    def __init__(self, emission, transition):
        """
        Initialize class
        :param emission: emission dataframe containing probabilities
        :param transition: transition dataframe containing probabilities
        """
        self.emission = emission
        self.transition = transition
        self.tags = emission.index  # tags
        self.start_tag = 'Start'  # start tag
        self.end_tag = 'End'  # end tag

    def create_trellis_tables(self, sequence):
        """
        Create empty viterbi_variable, back_pointer and hidden_state variables
        :param sequence: the sentence to be decoded
        :return: empty dataframe for viterbi variable, back pointer variable and empty list for hidden states
        """

        len_seq = len(sequence)
        v = pd.DataFrame(0.0, columns=range(len_seq), index=self.tags)
        b = pd.DataFrame(None, columns=range(len_seq), index=self.tags)
        y = [None] * len_seq
        return v, b, y

    def score(self, word_in_sentence, current_tag, previous_tag):
        """
        Calculate the score for each word in sentence based on current and previous words
        :param word_in_sentence: The word in sentence for which we want to calculate the score (m)
        :param current_tag: The current word in sentence (k)
        :param previous_tag: The previous word in sentence (k')
        :return: (emission + transition) score
        """

        if current_tag == 'End':
            # there is no emission score for end word
            em_score = 0.0
        else:
            # get the emission score
            em_score = self.emission.loc[current_tag][word_in_sentence]

        # get the transition score
        tr_score = self.transition.loc[previous_tag][current_tag]

        # add the scores
        score = em_score + tr_score

        return score

    def trellis(self, sequence):
        """
        Calculate the trellis tables for sequence
        :param sequence: the sentence to be decoded
        :return: trellis tables v, b, y
        """

        # instantiate trellis tables
        v, b, y = self.create_trellis_tables(sequence)
        m = len(sequence)  # length of sentence

        # get the list of tags for easy reference
        tags = self.tags

        # start
        for i in tags:
            v.loc[i][0] = self.score(sequence[0], i, self.start_tag)
            b.loc[i][0] = self.start_tag

        # sentence
        for i in range(1, m):
            for j in tags:
                scores = [self.score(sequence[i], j, x) + v.loc[x][i - 1] for x in tags]
                v.loc[j][i] = max(scores)
                b.loc[j][i] = tags[np.argmax(scores)]

        # end
        y[m - 1] = tags[np.argmax([self.score(sequence[m - 1], self.end_tag, x) + v.loc[x][m - 1] for x in tags])]
        for i in range(m - 2, -1, -1):
            y[i] = b.loc[y[i + 1]][i + 1]

        # note: due to time constraints, if running on large number of sequences return only the decoding
        return y
        # note: change return statement to get the trellis tables along with decoded sequence
        # return v, b, y
