

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
    paddle.set_device('cpu')

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

    beam_size = 5
    model = paddle.Model(
        Seq2SeqAttnInferModel(
            vocab_size,
            hidden_size,
            hidden_size,
            num_layers,
            bos_id=bos_id,
            eos_id=eos_id,
            beam_size=beam_size,
            max_out_len=256))
    model.prepare()
    model.load('couplet_models/final')

    while 1:
        strr = input('请输入上联：')

        src, src_pos, src_len = convert_example(strr, vocab)
        src = paddle.to_tensor([src], dtype='int64')
        src_pos = paddle.to_tensor([src_pos], dtype='int64')
        src_len = paddle.to_tensor([src_len], dtype='int64')

        finished_seq = model.network.forward(src, src_pos, src_len)
        finished_seq = finished_seq[:, :, np.newaxis] if len(finished_seq.shape) == 2 else finished_seq
        finished_seq = np.transpose(finished_seq, [0, 2, 1])
        beam = finished_seq[0][0]
        id_list = np.array(post_process_seq(beam, bos_id, eos_id))
        word_list_s = [trg_idx2word[id[0]] for id in id_list]
        result = "".join(word_list_s)

        print('下联：', result)
        print()
