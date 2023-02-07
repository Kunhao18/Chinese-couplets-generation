
import paddle
from functools import partial
import jieba.posseg as pseg
import numpy as np
from paddlenlp.data import Pad


pos_idx = {'a': 1, 'ad': 2, 'ag': 3, 'an': 4,
           'b': 5, 'c': 6, 'd': 7, 'df': 8,
           'dg': 9, 'e': 10, 'f': 11, 'g': 12,
           'h': 13, 'i': 14, 'j': 15, 'k': 16,
           'l': 17, 'm': 18, 'mg': 19, 'mq': 20,
           'n': 21, 'ng': 22, 'nr': 23, 'nrfg': 24,
           'nrt': 25, 'ns': 26, 'nt': 27, 'nz': 28,
           'o': 29, 'p': 30, 'q': 31, 'r': 32,
           'rg': 33, 'rr': 34, 'rz': 35, 's': 36,
           't': 37, 'tg': 38, 'u': 39, 'ud': 40,
           'ug': 41, 'uj': 42, 'ul': 43, 'uv': 44,
           'uz': 45, 'v': 46, 'vd': 47, 'vg': 48,
           'vi': 49, 'vn': 50, 'vq': 51, 'x': 52,
           'y': 53, 'yg': 54, 'z': 55, 'zg': 56,
           'in': 57, 'bg': 58}


def convert_example(example, vocab):
    bos_id = vocab[vocab.bos_token]
    eos_id = vocab[vocab.eos_token]

    source_list = [c for c in example]
    source = [bos_id] + vocab.to_indices(source_list) + [eos_id]

    # jieba.enable_paddle()
    source_join = ''.join(source_list)
    source_words = pseg.cut(source_join)

    source_pos = []
    for word in source_words:
        for i in range(len(word.word)):
            source_pos.append(pos_idx[word.flag])

    source_pos = [0] + source_pos + [0]

    return source, source_pos, len(source)


def convert_example_score(example, vocab):
    pad_id = vocab[vocab.eos_token]
    bos_id = vocab[vocab.bos_token]
    eos_id = vocab[vocab.eos_token]

    source_list = [c for c in example[0]]
    target_list = [c for c in example[1]]
    source = [bos_id] + vocab.to_indices(source_list) + [eos_id]
    target = [bos_id] + vocab.to_indices(target_list) + [eos_id]

    # jieba.enable_paddle()
    source_join = ''.join(source_list)
    target_join = ''.join(target_list)
    source_words = pseg.cut(source_join)
    target_words = pseg.cut(target_join)

    source_pos = []
    target_pos = []
    for word in source_words:
        for i in range(len(word.word)):
            source_pos.append(pos_idx[word.flag])
    source_pos = [0] + source_pos + [0]

    for word in target_words:
        for i in range(len(word.word)):
            target_pos.append(pos_idx[word.flag])
    target_pos = [0] + target_pos + [0]

    return source, target, source_pos, target_pos
