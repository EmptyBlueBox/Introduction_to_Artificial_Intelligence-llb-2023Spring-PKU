import os
import random
import string
import nltk
import pickle


class basedataset():
    def __init__(self, mode, maxlen=None):
        assert mode in ['train', 'test', 'dev']
        self.root = './SST_2/'+mode+'.tsv'
        f = open(self.root, 'r', encoding='utf-8')
        L = f.readlines()
        self.data = [x.strip().split('\t') for x in L]
        if maxlen is not None:
            self.data = self.data[:maxlen]
        self.len = len(self.data)
        self.count = 0

    def tokenize(self, text):
        cleaned_tokens = []
        tokens = nltk.tokenize.word_tokenize(text.lower())
        for token in tokens:
            if token in nltk.corpus.stopwords.words('english'):
                continue
            else:
                all_punct = True
                for char in token:
                    if char not in string.punctuation:
                        all_punct = False
                        break
                if not all_punct:
                    cleaned_tokens.append(token)
        return cleaned_tokens

    def __getitem__(self, index):
        text, label = self.data[index]
        text = text.strip()
        text = self.tokenize(text)
        return text, int(label)

    def get(self, index):
        text, label = self.data[index]
        return text, int(label)


def traindataset():
    return basedataset('train')


def minitraindataset():
    return basedataset('train', maxlen=1200)


def testdataset():
    return basedataset('dev')


def validationdataset():
    return basedataset('dev')


if __name__ == '__main__':
    pass
