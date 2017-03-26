from itertools import chain
from collections import Counter
import os
import numpy as np

from utils import invert_dict

def parse_sentence(sentence):
    parsed_sentence = []
    for entity in sentence.split():
        token, tag = entity.split('/')[-2:]
        parsed_sentence.append((token, tag))
    return parsed_sentence


def trim_dictionary(dictionary, cutoff=5):
    for key, value in dictionary.items():
        if value <= cutoff:
            dictionary.pop(key)
    return dictionary


def parse_sentences(dir_path):
    files = os.listdir(dir_path)
    sentences = []
    tags = set()
    words = Counter()
    for file_path in files:
        _file = open('{dir_path}/{file}'.format(
            file=file_path, dir_path=dir_path))
        for line in _file:
            sent = parse_sentence(line)
            if len(sent) < 2:
                continue
            sentences.append(sent)
            for word, tag in sent:
                tags.add(tag)
                words[word] += 1
    words = trim_dictionary(words)
    return sentences, tags, words


def seq_to_windows(words, tags, word_to_num, tag_to_num,
                   left=1, right=1):
    X = []
    y = []
    for i in range(len(words)-1):
        tagn = tag_to_num[tags[i]]
        idxs = [word_to_num[words[ii]]
                for ii in range(i - left, i + right + 1)]
        X.append(idxs)
        y.append(tagn)
    return np.array(X), np.array(y)


def pad_sequence(seq, left=1, right=1):
    return left*[('<s>', '<s>')] + seq + right*[("</s>", '</s>')]


def docs_to_windows(docs, word_to_num, tag_to_num, window_size=3):
    pad = (window_size - 1)/2
    docs = list(chain.from_iterable(
        [pad_sequence(seq, left=pad, right=pad) for seq in docs]
    ))
    words, tags = zip(*docs)
    words, tags = list(words), list(tags)
    for i, word in enumerate(words):
        if word not in word_to_num:
            words[i] = 'UUUNKKK'
    return seq_to_windows(words, tags, word_to_num,
                          tag_to_num, pad, pad)


def preprocess_data(dir_path='NKJP_1.2_nltk_POS', window_size=3):
    sentences, tagnames, dictionary = parse_sentences(dir_path)
    print '{} sentences loaded'.format(len(sentences))
    tagnames.update(['<s>', '</s>'])  # Add special tags
    num_to_tag = dict(enumerate(tagnames))
    tag_to_num = invert_dict(num_to_tag)

    dictionary.update(['UUUNKKK', '<s>', '</s>'])  # Add special tokens
    num_to_word = dict(enumerate(dictionary))
    word_to_num = invert_dict(num_to_word)

    X, y = docs_to_windows(sentences, word_to_num, tag_to_num, window_size)
    print '{} {}-word windows loaded'.format(len(X), window_size)
    print 'Shape of X is {}\nShape of y is {}'.format(X.shape, y.shape)
    return X, y, word_to_num, tag_to_num
