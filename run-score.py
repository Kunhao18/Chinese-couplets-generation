from functools import partial
import paddle
import scipy.stats
import paddle.nn.functional as F
from paddlenlp.data import Vocab
from paddlenlp.datasets import load_dataset
import numpy as np

from models import Seq2SeqAttnInferModel, CoupletJudge
from util_funcs import convert_example, convert_example_score

import socket
import ctypes
import struct

batch_size = 128


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


if __name__ == '__main__':
    train_ds, test_ds = load_dataset('couplet', splits=('train', 'test'))

    vocab = Vocab.load_vocabulary(**train_ds.vocab_info)
    trg_idx2word = vocab.idx_to_token
    vocab_size = len(vocab)

    pad_id = vocab[vocab.eos_token]
    bos_id = vocab[vocab.bos_token]
    eos_id = vocab[vocab.eos_token]

    batch_size = 128
    num_layers = 2
    hidden_size = 256
    model_path = './couplet_models'

    model_score = paddle.Model(
        CoupletJudge(vocab_size, hidden_size, hidden_size,
                     num_layers, pad_id))
    model_score.prepare()
    model_score.load('couplet_models/final')

    while 1:
        strr_pre = input('请输入上联：')
        strr_pro = input('请输入下联：')

        judge = (strr_pre, strr_pro)
        (source_words, target_words) = judge
        source, target, source_pos, target_pos = convert_example_score(judge, vocab)

        source = paddle.to_tensor([source], dtype='int64')
        target = paddle.to_tensor([target], dtype='int64')
        source_pos = paddle.to_tensor([source_pos], dtype='int64')
        target_pos = paddle.to_tensor([target_pos], dtype='int64')

        src_length = len(source[0])
        vocab_size = len(vocab)
        src_length = paddle.to_tensor([src_length], dtype='int64')

        predict = model_score.network.forward(source, source_pos, src_length, target[:, :1], target_pos[:, :1])
        prob = []
        for i in range(len(target[0]) - 2):
            predict = model_score.network.forward(source, source_pos, src_length, target[:, :i + 1], target_pos[:, :i + 1])
            for j in range(len(predict[:, -1][0]) - 1):
                if trg_idx2word[j] == target_words[i]:
                    predict_sf = F.softmax(predict[:, -1][0])
                    prob.append(predict_sf[j])

        score = np.mean(np.array(prob))
        score *= 100
        result = format(score, ".2f")
        print('您的得分：', result)
        print()
