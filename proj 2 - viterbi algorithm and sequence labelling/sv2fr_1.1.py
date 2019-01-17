# -*- coding: utf-8 -*-
"""
Created on 28 Oct 2018

@author: Sri
"""

from sv2fr_viterbi import *

# input transition probabilities
transition_probabilities = {'H': [0.6, 0.4, 0.2], 'L': [0.4, 0.4, 0.5], 'End': [0.0, 0.2, 0.3]}
transition = pd.DataFrame(data=transition_probabilities, index=['Start', 'H', 'L'])
transition = transition.apply(lambda x: np.log(x))
print(transition)

# input emission probabilities
emission_probabilities = {'A': [0.2, 0.3], 'C': [0.3, 0.2], 'G': [0.3, 0.2], 'T': [0.2, 0.3]}
emission = pd.DataFrame(data=emission_probabilities, index=['H', 'L'])
emission = emission.apply(lambda x: np.log(x))
print(emission)

# input sequences
seq_str = [list('GCACTG')]
sequences = pd.DataFrame(columns=[0], index=range(len(seq_str)))
sequences[0] = seq_str
print(sequences)

# call viterbi
# note: in trellis method, modify the return method to get the trellis tables
viterbi = Viterbi(emission, transition)
for sequence in sequences[0]:
    v, b, y = viterbi.trellis(sequence)
    print(v)
    print(b)
    print(y)
