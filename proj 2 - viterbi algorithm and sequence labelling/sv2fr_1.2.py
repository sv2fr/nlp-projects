# -*- coding: utf-8 -*-
"""
Created on 28 Oct 2018

@author: Sri
"""

from sv2fr_utils import *
from sv2fr_viterbi import *
from collections import Counter


# ------------------------ reading and processing data ------------------------ #


# read the train, dev and test sets
train = read_file_into_df('trn.pos')
print(train.shape)
train.head()
dev = read_file_into_df('dev.pos')
print(dev.shape)
dev.head()
test = read_file_into_df('tst.word')
print(test.shape)
test.head()

# split by space
train[1] = split_by_space(train[0])
train.head()
dev[1] = split_by_space(dev[0])
dev.head()
test[1] = split_by_space(test[0])
test.head()

# split by /
train[2], train[3] = split_by_slash(train[1])
train.head()
dev[2], dev[3] = split_by_slash(dev[1])
dev.head()

# get all words before Unk
train_all_words = get_all_words(train[2])
dev_all_words = get_all_words(dev[2])
test_all_words = get_all_words(test[1])

# choose cutoff frequency
# count the word frequency
train_freq_count = Counter(train_all_words)
# get the number of times each frequency is occurring
freq_counts = Counter(train_freq_count.values())
freq_counts.most_common(20)
cutoff_freq = 6

# replace all less frequent words with Unk in train
train[4] = replace_least_common_with_unk(train[2], train_freq_count, cutoff_freq)
train.head()

# get the train vocabulary
# get all words after replacing with Unk
train_all_words_unk = get_all_words(train[4])
# get the vocabulary and its size
train_freq_count_unk = Counter(train_all_words_unk)
train_vocab = set(train_freq_count_unk.keys())
len(train_vocab)

# replace words in dev not in train vocab with Unk
dev[4] = replace_non_vocab_words_with_unk(dev[2], train_vocab)
dev.head()
# replace words in test not in train vocab with Unk
test[2] = replace_non_vocab_words_with_unk(test[1], train_vocab)
test.head()


# ------------------------ creating probability tables ------------------------ #


# create dataframes for storing transition and emission probabilities
# get list of all tags
tags = get_all_tags(train[3])
print(tags)

# create dataframe to store transition probabilities
transition = get_zero_dataframe(['Start'] + tags, tags + ['End'])
print(transition.shape)
print(transition)

# create dataframe to store transition_smoothed probabilities
transition_smoothed = get_zero_dataframe(['Start'] + tags, tags + ['End'])
print(transition_smoothed.shape)
print(transition_smoothed)

# create dataframe to store emission probabilities
emission = get_zero_dataframe(tags, train_freq_count_unk.keys())
print(emission.shape)
print(emission)

# create dataframe to store emission_smoothed probabilities
emission_smoothed = get_zero_dataframe(tags, train_freq_count_unk.keys())
print(emission_smoothed.shape)
print(emission_smoothed)


# ------------------------ calculating transition probabilities ------------------------ #


# add start and end tags
add_start_end_tags(train[3])
add_start_end_tags(dev[3])
train.head()
dev.head()

# join all tags to a single list
train_tags_all = get_all_words(train[3])

# count frequency of each tag
train_tags_freq = Counter(train_tags_all)
print(train_tags_freq)

# make bi-gram tuples
bigram = list(zip(train_tags_all, train_tags_all[1:]))

# count tuple frequencies
train_tuple_freq = Counter(bigram)
print(train_tuple_freq)

# iterate through transition tables and fill in probabilities
for row in transition.index:
    for column in transition.columns:
        transition.loc[row][column] = train_tuple_freq[(row, column)] / train_tags_freq[row]
print(transition)

beta = 0.001  # the best value based on CV
N = len(tags)
for row in transition_smoothed.index:
    for column in transition_smoothed.columns:
        transition_smoothed.loc[row][column] = (train_tuple_freq[(row, column)] + beta) / (train_tags_freq[row] + N * beta)
print(transition_smoothed)


# verify that sum of transition probabilities for each tag is equal to one
# NOTE: because floating point arithmetic is not precise in computers we are not getting values equal to 1
for row in transition.index:
    prob_sum = 0
    for column in transition.columns:
        prob_sum += transition.loc[row][column]
    print(row, prob_sum)
for row in transition_smoothed.index:
    prob_sum = 0
    for column in transition_smoothed.columns:
        prob_sum += transition_smoothed.loc[row][column]
    print(row, prob_sum)

# write to file
write_to_csv(transition, 'sv2fr-tprob.txt')
write_to_csv(transition_smoothed, 'sv2fr-tprob-smoothed.txt')


# ------------------------ calculating emission probabilities ------------------------ #


# get the train tags frequency with out start and end tokens
train_tags_freq.pop('Start')
train_tags_freq.pop('End')
print(train_tags_freq)

# remove the start and end tokens from tags
remove_start_end_tags(train[3])
remove_start_end_tags(dev[3])
train.head()
dev.head()

# create list of tuples of <y, x> pairs
# get list of all tags
train_all_tags = get_all_words(train[3])
# get list of all words
train_all_words = get_all_words(train[4])
# create tuple of tag-word combo
list_tag_word = list(zip(train_all_tags, train_all_words))

# count freq
list_tag_word_freq = Counter(list_tag_word)
len(list_tag_word_freq)

# iterate through emission tables and fill in probabilities
for row in emission.index:
    for column in emission.columns:
        emission.loc[row][column] = list_tag_word_freq[(row, column)] / train_tags_freq[row]
print(emission)

alpha = 0.001  # the best value based on CV
V = len(set(train_all_words))  # size of vocab
for row in emission_smoothed.index:
    for column in emission_smoothed.columns:
        emission_smoothed.loc[row][column] = (list_tag_word_freq[(row, column)] + alpha) / (train_tags_freq[row] + V * alpha)
print(emission_smoothed)


# verify that sum of emission probabilities for each tag is equal to one
# NOTE: because floating point arithmetic is not precise in computers we are not getting values equal to 1
for row in emission.index:
    prob_sum = 0
    for column in emission.columns:
        prob_sum += emission.loc[row][column]
    print(row, prob_sum)
for row in emission_smoothed.index:
    prob_sum = 0
    for column in emission_smoothed.columns:
        prob_sum += emission_smoothed.loc[row][column]
    print(row, prob_sum)

# write to file
write_to_csv(emission, 'sv2fr-eprob.txt')
write_to_csv(emission_smoothed, 'sv2fr-eprob-smoothed.txt')


# ------------------------ calculating log probabilities ------------------------ #


# convert smoothed probabilities to log probabilities
for row in emission_smoothed.index:
    for column in emission_smoothed.columns:
        emission_smoothed.loc[row][column] = np.log(emission_smoothed.loc[row][column])
print(emission_smoothed)

for row in transition_smoothed.index:
    for column in transition_smoothed.columns:
        transition_smoothed.loc[row][column] = np.log(transition_smoothed.loc[row][column])
print(transition_smoothed)


# ------------------------ implementing viterbi on dev ------------------------ #


viterbi = Viterbi(emission_smoothed, transition_smoothed)

dev[5] = dev[4]
dev[5] = dev[4].apply(lambda x: viterbi.trellis(x))
dev.head()

# display the decoding for first 10 rows to compare
# display = pd.DataFrame(columns=[0], index=range(5))
# display[0] = dev[5][0:5]
# display[1] = dev[3][0:5]
# display.columns = ['predicted', 'actual']
# print(display)


# ------------------------ accuracy on dev ------------------------ #


predicted_list = get_all_words(dev[5])
actual_list = get_all_words(dev[3])
matches = count_matches(predicted_list, actual_list)
accuracy = matches / len(actual_list)
print(accuracy)

# average accuracy per sentence
# dev[6] = 0
# for i in range(0, len(dev.index) - 1):
#     dev[6][i] = count_matches(dev[3][i], dev[5][i])
# count = dev[6].sum(axis=0)
# count/len(dev.index)


# ------------------------ decoding on test ------------------------ #


test[3] = test[2]
test[3] = test[2].apply(lambda x: viterbi.trellis(x))
test.head()

# write to file
test[4] = test[3]
for i in range(0, len(test.index) - 1):
    test[4][i] = combine_by_slash(test[1][i], test[3][i])
test[4].to_frame().to_csv('sv2fr-viterbi.txt', sep=' ', header=None, index=None, quoting=csv.QUOTE_NONE, escapechar=' ')