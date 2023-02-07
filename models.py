
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Seq2SeqEncoder(nn.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(Seq2SeqEncoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, embed_dim)
        self.posemb = nn.Embedding(59, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # dropout=0.2 if num_layers > 1 else 0)
            dropout=0)

    def forward(self, sequence, pos, sequence_length):
        inputs = self.embedder(sequence)
        inputs2 = self.posemb(pos)
        # print("shape:", inputs.shape, inputs2.shape)
        inputs += inputs2
        encoder_output, encoder_state = self.lstm(
            inputs, sequence_length=sequence_length)

        # encoder_output [128, 18, 256]  [batch_size, time_steps, hidden_size]
        # encoder_state (tuple) - 最终状态,一个包含h和c的元组。 [2, 128, 256] [2, 128, 256] [num_layers * num_directions, batch_size, hidden_size]
        return encoder_output, encoder_state


class AttentionLayer(nn.Layer):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.input_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(self, hidden, encoder_output, encoder_padding_mask):
        encoder_output = self.input_proj(encoder_output)
        attn_scores = paddle.matmul(
            paddle.unsqueeze(hidden, [1]), encoder_output, transpose_y=True)
        # print('attention score', attn_scores.shape) #[128, 1, 18]

        if encoder_padding_mask is not None:
            attn_scores = paddle.add(attn_scores, encoder_padding_mask)

        attn_scores = F.softmax(attn_scores)
        attn_out = paddle.squeeze(
            paddle.matmul(attn_scores, encoder_output), [1])
        # print('1 attn_out', attn_out.shape) #[128, 256]

        attn_out = paddle.concat([attn_out, hidden], 1)
        # print('2 attn_out', attn_out.shape) #[128, 512]

        attn_out = self.output_proj(attn_out)
        # print('3 attn_out', attn_out.shape) #[128, 256]
        return attn_out


class Seq2SeqDecoderCell(nn.RNNCellBase):
    def __init__(self, num_layers, input_size, hidden_size):
        super(Seq2SeqDecoderCell, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.lstm_cells = nn.LayerList([
            nn.LSTMCell(
                input_size=input_size + hidden_size if i == 0 else hidden_size,
                hidden_size=hidden_size) for i in range(num_layers)
        ])

        self.attention_layer = AttentionLayer(hidden_size)

    def forward(self,
                step_input,
                states,
                encoder_output,
                encoder_padding_mask=None):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = paddle.concat([step_input, input_feed], 1)
        for i, lstm_cell in enumerate(self.lstm_cells):
            out, new_lstm_state = lstm_cell(step_input, lstm_states[i])
            # step_input = self.dropout(out)
            step_input = out
            new_lstm_states.append(new_lstm_state)
        out = self.attention_layer(step_input, encoder_output,
                                   encoder_padding_mask)
        return out, [new_lstm_states, out]


class Seq2SeqDecoder(nn.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(Seq2SeqDecoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, embed_dim)
        self.posemb = nn.Embedding(59, embed_dim)
        self.lstm_attention = nn.RNN(
            Seq2SeqDecoderCell(num_layers, embed_dim, hidden_size))
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, trg, pos, decoder_initial_states, encoder_output,
                encoder_padding_mask):
        inputs = self.embedder(trg)
        inputs2 = self.posemb(pos)
        # print("shape:", inputs.shape, inputs2.shape)
        inputs += inputs2

        decoder_output, _ = self.lstm_attention(
            inputs,
            initial_states=decoder_initial_states,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        predict = self.output_layer(decoder_output)

        return predict


class Seq2SeqAttnModel(nn.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 eos_id=1):
        super(Seq2SeqAttnModel, self).__init__()
        self.hidden_size = hidden_size
        self.eos_id = eos_id
        self.num_layers = num_layers
        self.INF = 1e9
        self.encoder = Seq2SeqEncoder(vocab_size, embed_dim, hidden_size,
                                      num_layers)
        self.decoder = Seq2SeqDecoder(vocab_size, embed_dim, hidden_size,
                                      num_layers)

    def forward(self, src, src_pos, src_length, trg, trg_pos):
        # encoder_output 各时刻的输出h
        # encoder_final_state 最后时刻的输出h，和记忆信号c
        encoder_output, encoder_final_state = self.encoder(src, src_pos, src_length)
        print('encoder_output shape', encoder_output.shape)  # [128, 18, 256]  [batch_size,time_steps,hidden_size]
        print('encoder_final_states shape', encoder_final_state[0].shape, encoder_final_state
        [1].shape)  # [2, 128, 256] [2, 128, 256] [num_lauers * num_directions, batch_size, hidden_size]

        # Transfer shape of encoder_final_states to [num_layers, 2, batch_size, hidden_size]？？？
        encoder_final_states = [
            (encoder_final_state[0][i], encoder_final_state[1][i])
            for i in range(self.num_layers)
        ]
        print('encoder_final_states shape', encoder_final_states[0][0].shape,
              encoder_final_states[0][1].shape)  # [128, 256] [128, 256]

        # Construct decoder initial states: use input_feed and the shape is
        # [[h,c] * num_layers, input_feed], consistent with Seq2SeqDecoderCell.states
        decoder_initial_states = [
            encoder_final_states,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]

        # Build attention mask to avoid paying attention on padddings
        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())
        print('src_mask shape', src_mask.shape)  # [128, 18]
        print(src_mask[0, :])

        encoder_padding_mask = (src_mask - 1.0) * self.INF
        print('encoder_padding_mask', encoder_padding_mask.shape)  # [128, 18]
        print(encoder_padding_mask[0, :])

        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])
        print('encoder_padding_mask', encoder_padding_mask.shape)  # [128, 1, 18]

        predict = self.decoder(trg, trg_pos, decoder_initial_states, encoder_output,
                               encoder_padding_mask)
        print('predict', predict.shape)  # [128, 17, 7931]

        return predict


class Seq2SeqAttnInferModel(Seq2SeqAttnModel):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256):
        self.bos_id = bos_id
        self.beam_size = beam_size
        self.max_out_len = max_out_len
        self.num_layers = num_layers
        super(Seq2SeqAttnInferModel, self).__init__(
            vocab_size, embed_dim, hidden_size, num_layers, eos_id)

        # Dynamic decoder for inference
        self.beam_search_decoder = nn.BeamSearchDecoder(
            self.decoder.lstm_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.decoder.embedder,
            output_fn=self.decoder.output_layer)

    def forward(self, src, src_pos, src_length):
        encoder_output, encoder_final_state = self.encoder(src, src_pos, src_length)

        encoder_final_state = [
            (encoder_final_state[0][i], encoder_final_state[1][i])
            for i in range(self.num_layers)
        ]

        # Initial decoder initial states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # Build attention mask to avoid paying attention on paddings
        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())

        encoder_padding_mask = (src_mask - 1.0) * self.INF
        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])

        # Tile the batch dimension with beam_size
        encoder_output = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output, self.beam_size)
        encoder_padding_mask = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_padding_mask, self.beam_size)

        # Dynamic decoding with beam search
        seq_output, _ = nn.dynamic_decode(
            decoder=self.beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=self.max_out_len,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        return seq_output


class CoupletJudge(Seq2SeqAttnModel):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, eos_id):
        super(CoupletJudge, self).__init__(
            vocab_size, embed_dim, hidden_size, num_layers, eos_id)
        self.hidden_size = hidden_size
        self.eos_id = eos_id
        self.num_layers = num_layers
        self.INF = 1e9
        self.encoder = Seq2SeqEncoder(vocab_size, embed_dim, hidden_size,
                                      num_layers)
        self.decoder = Seq2SeqDecoder(vocab_size, embed_dim, hidden_size,
                                      num_layers)

    def forward(self, src, src_pos, src_length, trg, trg_pos):
        #print("this")
        encoder_output, encoder_final_state = self.encoder(src, src_pos, src_length)
        #print("encoder output: ", encoder_output)
        #print("encode ok")
        encoder_final_state = [
            (encoder_final_state[0][i], encoder_final_state[1][i])
            for i in range(self.num_layers)
        ]

        # Initial decoder initial states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        #print("state ok")
        # Build attention mask to avoid paying attention on paddings
        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())

        encoder_padding_mask = (src_mask - 1.0) * self.INF
        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])

        #print("here")
        predict = self.decoder(trg, trg_pos, decoder_initial_states, encoder_output,
                               encoder_padding_mask)
        #print("last: ", predict)
        return predict
