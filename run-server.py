

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

    model_score = paddle.Model(
        CoupletJudge(vocab_size, hidden_size, hidden_size,
                     num_layers, pad_id))
    model_score.prepare()
    model_score.load('couplet_models/final')

    HOST = '127.0.0.1'
    PORT = 5234
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()

    print("server start.")

    while 1:
        conn, addr = s.accept()
        print('connect by', addr)
        while 1:
            try:
                data = conn.recv(256)
                code = struct.unpack('i', data[:4])
                print(code)
            except ConnectionResetError:
                print(addr, 'disconnect')
                break

            if code[0] == 1:
                i = struct.unpack('i', data[4: 8])
                strr = bytes.decode(data[8: 8 + i[0] * 3])

                src, src_pos, src_len = convert_example(strr, vocab)
                src = paddle.to_tensor([src], dtype='int64')
                src_pos = paddle.to_tensor([src_pos], dtype='int64')
                src_len = paddle.to_tensor([src_len], dtype='int64')

                # inputs =
                finished_seq = model.network.forward(src, src_pos, src_len)
                finished_seq = finished_seq[:, :, np.newaxis] if len(
                    finished_seq.shape) == 2 else finished_seq
                finished_seq = np.transpose(finished_seq, [0, 2, 1])
                beam = finished_seq[0][0]
                id_list = np.array(post_process_seq(beam, bos_id, eos_id))
                word_list_s = [trg_idx2word[id[0]] for id in id_list]
                result = "".join(word_list_s)

                conn.send(str.encode(result), 0)
            elif code[0] == 0:
                offset = 4
                i = struct.unpack('i', data[offset: offset + 4])
                offset += 4
                strr_pre = bytes.decode(data[offset: offset + i[0] * 3])
                offset += (i[0] * 3)
                j = struct.unpack('i', data[offset: offset + 4])
                offset += 4
                strr_pro = bytes.decode(data[offset: offset + j[0] * 3])

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
                    predict = model_score.network.forward(source, source_pos, src_length, target[:, :i + 1],
                                                       target_pos[:, :i + 1])
                    for j in range(len(predict[:, -1][0]) - 1):
                        if trg_idx2word[j] == target_words[i]:
                            predict_sf = F.softmax(predict[:, -1][0])
                            prob.append(predict_sf[j])

                score = np.mean(np.array(prob))
                score *= 100
                # score *= 1000
                result = format(score, ".2f")
                print(result)
                conn.send(str.encode(result), 0)
                pass
