import os


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'trn-wiki.txt'))
        self.dev = self.tokenize(os.path.join(path, 'dev-wiki.txt'))
        self.test = self.tokenize(os.path.join(path, 'tst-wiki.txt'))

    def tokenize(self, path):
        """
        Tokenize text file and convert all words in file to tensor
        :param path: path to file
        :return: LongTensor of tokens
        """
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split()
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            doc_ids = []
            for line in f:
                sent = []
                words = line.split()
                for word in words:
                    sent.append(self.dictionary.word2idx[word])
                doc_ids.append(sent)
        return doc_ids
